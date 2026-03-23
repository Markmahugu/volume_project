(function () {
  const React = window.React;
  const ReactDOM = window.ReactDOM;
  const THREE = window.THREE;

  if (!React || !ReactDOM || !THREE || !THREE.OrbitControls) {
    const root = document.getElementById('root');
    root.textContent = 'Failed to load the React/Three.js runtime. Check network access to the CDN scripts in the browser console.';
    return;
  }

  const h = React.createElement;
  const initialFiles = Array.isArray(window.__INITIAL_FILES__) ? window.__INITIAL_FILES__ : [];

  function formatNumber(value) {
    return Number(value).toLocaleString();
  }

  function createPointsObject(payload, fallbackColorHex) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(payload.positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(payload.colors, 3));
    geometry.computeBoundingSphere();

    const material = new THREE.PointsMaterial({
      size: 0.08,
      sizeAttenuation: true,
      vertexColors: true,
      color: fallbackColorHex,
    });
    return new THREE.Points(geometry, material);
  }

  function App() {
    const viewerRef = React.useRef(null);
    const sceneRef = React.useRef(null);
    const cameraRef = React.useRef(null);
    const rendererRef = React.useRef(null);
    const controlsRef = React.useRef(null);
    const previewObjectRef = React.useRef(null);
    const roiObjectRef = React.useRef(null);
    const selectedObjectRef = React.useRef(null);
    const bboxRef = React.useRef(null);
    const markerGroupRef = React.useRef(null);
    const raycasterRef = React.useRef(new THREE.Raycaster());
    const pointerRef = React.useRef(new THREE.Vector2());
    const pickModeRef = React.useRef(false);

    const [files] = React.useState(initialFiles);
    const [selectedFile, setSelectedFile] = React.useState(initialFiles[0] || '');
    const [status, setStatus] = React.useState(initialFiles.length ? 'Opening the first dataset preview...' : 'No .ply or .pcd files found in the workspace root.');
    const [pickMode, setPickMode] = React.useState(false);
    const [pickedPoints, setPickedPoints] = React.useState([]);
    const [results, setResults] = React.useState(null);
    const [params, setParams] = React.useState({
      previewVoxel: 0.05,
      downsampleVoxel: 0.02,
      volumeVoxel: 0.02,
      dbscanEps: 0.08,
      dbscanMinPoints: 40,
      planeThreshold: 0.03,
      roiPaddingXY: 1.0,
      roiPaddingZ: 0.5,
    });

    React.useEffect(() => {
      pickModeRef.current = pickMode;
    }, [pickMode]);

    function clearObject(ref) {
      const object = ref.current;
      if (!object || !sceneRef.current) {
        return;
      }
      sceneRef.current.remove(object);
      if (object.geometry) {
        object.geometry.dispose();
      }
      if (object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach((material) => material.dispose());
        } else {
          object.material.dispose();
        }
      }
      ref.current = null;
    }

    function clearMarkers() {
      const group = markerGroupRef.current;
      if (!group) {
        return;
      }
      while (group.children.length) {
        const child = group.children[0];
        group.remove(child);
        if (child.geometry) {
          child.geometry.dispose();
        }
        if (child.material) {
          child.material.dispose();
        }
      }
    }

    function frameObject(object3d) {
      const bounds = new THREE.Box3().setFromObject(object3d);
      if (bounds.isEmpty()) {
        return;
      }
      const size = bounds.getSize(new THREE.Vector3());
      const center = bounds.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z, 1);
      const distance = maxDim * 1.7;
      cameraRef.current.position.set(center.x + distance, center.y + distance * 0.75, center.z + distance * 0.55);
      controlsRef.current.target.copy(center);
      controlsRef.current.update();
    }

    async function loadPreview(targetFile = selectedFile) {
      if (!targetFile) {
        setStatus('No point cloud file is available.');
        return;
      }

      setStatus(`Generating preview cloud for ${targetFile}...`);
      setResults(null);
      clearObject(roiObjectRef);
      clearObject(selectedObjectRef);
      clearObject(bboxRef);
      clearObject(previewObjectRef);
      clearMarkers();
      setPickedPoints([]);

      try {
        const response = await fetch(`/api/preview?input_path=${encodeURIComponent(targetFile)}&voxel_size=${params.previewVoxel}`);
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || 'Failed to load preview data.');
        }
        const preview = createPointsObject(payload, 0xd9d9d9);
        previewObjectRef.current = preview;
        sceneRef.current.add(preview);
        frameObject(preview);
        setStatus(`Preview opened: ${formatNumber(payload.point_count)} preview points from ${targetFile}.`);
      } catch (error) {
        setStatus(error.message || 'Preview loading failed.');
      }
    }

    async function analyzeSelection() {
      if (pickedPoints.length < 3) {
        setStatus('Pick at least 3 points on the sand pile before analysis.');
        return;
      }

      setStatus('Running backend segmentation and volume estimation...');
      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input_path: selectedFile,
            picked_points: pickedPoints,
            downsample_voxel: Number(params.downsampleVoxel),
            volume_voxel: Number(params.volumeVoxel),
            dbscan_eps: Number(params.dbscanEps),
            dbscan_min_points: Number(params.dbscanMinPoints),
            plane_threshold: Number(params.planeThreshold),
            roi_padding_xy: Number(params.roiPaddingXY),
            roi_padding_z: Number(params.roiPaddingZ),
          }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || 'Analysis failed.');
        }

        clearObject(roiObjectRef);
        clearObject(selectedObjectRef);
        clearObject(bboxRef);

        const roiObject = createPointsObject(payload.roi_cloud, 0x9e9e9e);
        const selectedObject = createPointsObject(payload.selected_cloud, 0x37d871);
        roiObjectRef.current = roiObject;
        selectedObjectRef.current = selectedObject;
        sceneRef.current.add(roiObject);
        sceneRef.current.add(selectedObject);

        const min = new THREE.Vector3(...payload.bbox.min);
        const max = new THREE.Vector3(...payload.bbox.max);
        const helper = new THREE.Box3Helper(new THREE.Box3(min, max), 0xff5f45);
        bboxRef.current = helper;
        sceneRef.current.add(helper);
        frameObject(selectedObject);

        setResults([
          ['Clusters detected', String(payload.cluster_count)],
          ['Selected cluster', String(payload.selected_cluster_index)],
          ['ROI points', formatNumber(payload.roi_points)],
          ['Non-ground points', formatNumber(payload.object_points)],
          ['Selected points', formatNumber(payload.selected_points)],
          ['BBox volume', `${payload.bbox_volume_m3.toFixed(4)} m³`],
          ['Voxel volume', `${payload.voxel_volume_m3.toFixed(4)} m³`],
        ]);
        setStatus(`Analysis complete. Estimated sand volume: ${payload.voxel_volume_m3.toFixed(4)} m³.`);
      } catch (error) {
        setStatus(error.message || 'Analysis failed.');
      }
    }

    React.useEffect(() => {
      const viewerEl = viewerRef.current;
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x10131a);
      scene.fog = new THREE.Fog(0x10131a, 35, 140);
      sceneRef.current = scene;

      const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);
      camera.position.set(12, 10, 12);
      cameraRef.current = camera;

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(window.devicePixelRatio);
      rendererRef.current = renderer;
      viewerEl.appendChild(renderer.domElement);

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.target.set(0, 0, 0);
      controlsRef.current = controls;

      raycasterRef.current.params.Points.threshold = 0.35;

      scene.add(new THREE.AmbientLight(0xffffff, 1.2));
      const directional = new THREE.DirectionalLight(0xfff2d9, 1.4);
      directional.position.set(10, 18, 12);
      scene.add(directional);
      scene.add(new THREE.GridHelper(80, 80, 0x7a5b44, 0x40352a));
      scene.add(new THREE.AxesHelper(3));

      const markerGroup = new THREE.Group();
      markerGroupRef.current = markerGroup;
      scene.add(markerGroup);

      function resize() {
        const width = viewerEl.clientWidth;
        const height = viewerEl.clientHeight;
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      }

      function animate() {
        controls.update();
        renderer.render(scene, camera);
        requestAnimationFrame(animate);
      }

      function onClick(event) {
        if (!pickModeRef.current || !previewObjectRef.current) {
          return;
        }
        const rect = renderer.domElement.getBoundingClientRect();
        pointerRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointerRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycasterRef.current.setFromCamera(pointerRef.current, camera);
        const intersections = raycasterRef.current.intersectObject(previewObjectRef.current);
        if (intersections.length === 0) {
          return;
        }
        const point = intersections[0].point.clone();
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(0.18, 18, 18),
          new THREE.MeshStandardMaterial({ color: 0xff6b35, emissive: 0x7a1b00 })
        );
        marker.position.copy(point);
        markerGroup.add(marker);
        setPickedPoints((previous) => previous.concat([[point.x, point.y, point.z]]));
      }

      resize();
      animate();
      renderer.domElement.addEventListener('click', onClick);
      window.addEventListener('resize', resize);

      if (initialFiles.length > 0) {
        loadPreview(initialFiles[0]);
      }

      return () => {
        renderer.domElement.removeEventListener('click', onClick);
        window.removeEventListener('resize', resize);
        viewerEl.removeChild(renderer.domElement);
      };
    }, []);

    function updateParam(name, value) {
      setParams((previous) => ({ ...previous, [name]: value }));
    }

    function renderInput(label, name, type, step, min) {
      return h('label', { key: name }, [
        label,
        h('input', {
          value: params[name],
          type,
          step,
          min,
          onChange: (event) => updateParam(name, event.target.value),
        }),
      ]);
    }

    function handleClearPicks() {
      clearMarkers();
      setPickedPoints([]);
      setStatus('Picked points cleared.');
    }

    return h('div', { className: 'layout' }, [
      h('aside', { className: 'sidebar', key: 'sidebar' }, [
        h('div', { key: 'hero' }, [
          h('p', { className: 'eyebrow' }, 'Construction Monitoring'),
          h('h1', null, 'Sand Volume Estimation'),
          h('p', { className: 'lede' }, 'React-driven browser workflow for previewing a point cloud, clicking seed points on the sand pile, and estimating volume only inside the selected region.'),
        ]),
        h('section', { className: 'panel', key: 'dataset' }, [
          h('h2', null, 'Dataset'),
          h('label', null, [
            'Point cloud',
            h('select', {
              value: selectedFile,
              onChange: (event) => {
                setSelectedFile(event.target.value);
                setStatus(`Selected dataset: ${event.target.value}. Click Load Preview.`);
              },
            }, files.length ? files.map((fileName) => h('option', { key: fileName, value: fileName }, fileName)) : h('option', { value: '' }, 'No point cloud files found')),
          ]),
          h('label', null, [
            'Preview voxel size',
            h('input', {
              type: 'number',
              min: '0.01',
              step: '0.01',
              value: params.previewVoxel,
              onChange: (event) => updateParam('previewVoxel', event.target.value),
            }),
          ]),
          h('button', { className: 'primary', onClick: () => loadPreview(selectedFile), disabled: !selectedFile }, 'Load Preview'),
        ]),
        h('section', { className: 'panel', key: 'parameters' }, [
          h('h2', null, 'Analysis Parameters'),
          h('div', { className: 'grid' }, [
            renderInput('Downsample voxel', 'downsampleVoxel', 'number', '0.005', '0.005'),
            renderInput('Volume voxel', 'volumeVoxel', 'number', '0.005', '0.005'),
            renderInput('DBSCAN eps', 'dbscanEps', 'number', '0.01', '0.01'),
            renderInput('DBSCAN min points', 'dbscanMinPoints', 'number', '1', '5'),
            renderInput('Plane threshold', 'planeThreshold', 'number', '0.005', '0.005'),
            renderInput('ROI padding XY', 'roiPaddingXY', 'number', '0.1', '0.1'),
            renderInput('ROI padding Z', 'roiPaddingZ', 'number', '0.1', '0.1'),
          ]),
        ]),
        h('section', { className: 'panel', key: 'picking' }, [
          h('h2', null, 'Picking'),
          h('p', { className: 'hint' }, 'Enable pick mode, then click at least 3 points on the sand pile in the viewer.'),
          h('div', { className: 'actions' }, [
            h('button', { onClick: () => { const next = !pickMode; setPickMode(next); setStatus(next ? 'Pick mode enabled. Click points on the sand pile.' : 'Pick mode disabled.'); } }, pickMode ? 'Disable Pick Mode' : 'Enable Pick Mode'),
            h('button', { className: 'ghost', onClick: handleClearPicks }, 'Clear Picks'),
          ]),
          h('div', { id: 'pickCount' }, `Picked points: ${pickedPoints.length}`),
          h('ol', { id: 'pickList' }, pickedPoints.map((point, index) => h('li', { key: `${index}-${point.join('-')}` }, `${index + 1}: ${point.map((value) => Number(value).toFixed(2)).join(', ')}`))),
          h('button', { className: 'primary', onClick: analyzeSelection, disabled: pickedPoints.length < 3 }, 'Analyze Selected Region'),
        ]),
        h('section', { className: 'panel results-panel', key: 'results' }, [
          h('h2', null, 'Results'),
          h('div', { id: 'status' }, status),
          results ? h('dl', null, results.flatMap(([label, value]) => [h('dt', { key: `${label}-label` }, label), h('dd', { key: `${label}-value` }, value)])) : null,
        ]),
      ]),
      h('main', { className: 'viewer-shell', key: 'viewer-shell' }, [
        h('div', { className: 'viewer-toolbar' }, [
          h('span', null, 'Left drag: orbit'),
          h('span', null, 'Right drag: pan'),
          h('span', null, 'Wheel: zoom'),
          h('span', null, 'Pick mode: click points on the sand pile'),
        ]),
        h('div', { id: 'viewer', ref: viewerRef }),
      ]),
    ]);
  }

  ReactDOM.createRoot(document.getElementById('root')).render(h(App));
})();
