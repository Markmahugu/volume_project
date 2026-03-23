(() => {
  const canvas = document.getElementById('viewerCanvas');
  const viewerStage = document.querySelector('.viewer-stage');
  const fileSelect = document.getElementById('fileSelect');
  const loadPreviewButton = document.getElementById('loadPreviewButton');
  const navigateModeButton = document.getElementById('navigateModeButton');
  const pointPickModeButton = document.getElementById('pointPickModeButton');
  const boxPickModeButton = document.getElementById('boxPickModeButton');
  const clearPicksButton = document.getElementById('clearPicksButton');
  const analyzeButton = document.getElementById('analyzeButton');
  const selectionModeLabel = document.getElementById('selectionModeLabel');
  const pickCount = document.getElementById('pickCount');
  const pickList = document.getElementById('pickList');
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  const initialFiles = Array.isArray(window.__INITIAL_FILES__) ? window.__INITIAL_FILES__ : [];

  if (!canvas || !viewerStage || !fileSelect || !loadPreviewButton || !clearPicksButton || !analyzeButton || !pickCount || !pickList || !status || !results) {
    return;
  }

  const inputs = {
    previewVoxel: document.getElementById('previewVoxel'),
    downsampleVoxel: document.getElementById('downsampleVoxel'),
    volumeVoxel: document.getElementById('volumeVoxel'),
    dbscanEps: document.getElementById('dbscanEps'),
    dbscanMinPoints: document.getElementById('dbscanMinPoints'),
    minClusterSize: document.getElementById('minClusterSize'),
    planeThreshold: document.getElementById('planeThreshold'),
    heightThreshold: document.getElementById('heightThreshold'),
    roiPaddingXY: document.getElementById('roiPaddingXY'),
    roiPaddingZ: document.getElementById('roiPaddingZ'),
  };

  const ctx = canvas.getContext('2d');
  const state = {
    yaw: -0.9,
    pitch: 0.55,
    zoom: 1,
    panX: 0,
    panY: 0,
    dragging: false,
    dragButton: 0,
    dragStartX: 0,
    dragStartY: 0,
    lastX: 0,
    lastY: 0,
    interactionMode: 'navigate',
    selectionKind: 'polygon',
    selectionBounds: null,
    preview: null,
    ground: null,
    filtered: null,
    selected: null,
    voxels: null,
    bbox: null,
    pickedPoints: [],
    projectedPreview: [],
    selectionBox: null,
    currentFile: initialFiles[0] || fileSelect.value || '',
  };

  function setStatus(message, isError = false) {
    status.textContent = message;
    status.style.color = isError ? '#b42318' : '#6e6254';
  }

  function updateResults(entries = []) {
    results.innerHTML = ''; 
    for (const [label, value] of entries) {
      const dt = document.createElement('dt');
      dt.textContent = label;
      const dd = document.createElement('dd');
      dd.textContent = value;
      results.append(dt, dd);
    }
  }

  function updateInteractionUI() {
    const isNavigate = state.interactionMode === 'navigate';
    const isPoint = state.interactionMode === 'point';
    const isBox = state.interactionMode === 'box';

    if (navigateModeButton) {
      navigateModeButton.classList.toggle('is-active', isNavigate);
    }
    if (pointPickModeButton) {
      pointPickModeButton.classList.toggle('is-active', isPoint);
    }
    if (boxPickModeButton) {
      boxPickModeButton.classList.toggle('is-active', isBox);
    }
    viewerStage.classList.toggle('mode-point', isPoint);
    viewerStage.classList.toggle('mode-box', isBox);
    viewerStage.classList.toggle('is-box-dragging', Boolean(state.selectionBox));

    if (!selectionModeLabel) {
      return;
    }
    selectionModeLabel.textContent = isNavigate
      ? 'Navigation mode active.'
      : isPoint
        ? 'Point mode active. Click around the sand footprint.'
        : 'Box mode active. Drag a 2D box; all cloud points inside that region will be included.';
  }

  function updatePickUI() {
    pickCount.textContent = `Picked points: ${state.pickedPoints.length}`;
    pickList.innerHTML = ''; 
    for (let index = 0; index < state.pickedPoints.length; index += 1) {
      const point = state.pickedPoints[index];
      const item = document.createElement('li');
      item.textContent = `${index + 1}: ${point.map((value) => value.toFixed(2)).join(', ')}`;
      pickList.appendChild(item);
    }
    analyzeButton.disabled = state.pickedPoints.length < 3 || !state.preview;
    updateInteractionUI();
  }

  function resizeCanvas() {
    const width = Math.max(1, viewerStage.clientWidth);
    const height = Math.max(1, viewerStage.clientHeight);
    canvas.width = Math.floor(width * window.devicePixelRatio);
    canvas.height = Math.floor(height * window.devicePixelRatio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    draw();
  }

  function rotatePoint(point) {
    const cosYaw = Math.cos(state.yaw);
    const sinYaw = Math.sin(state.yaw);
    const cosPitch = Math.cos(state.pitch);
    const sinPitch = Math.sin(state.pitch);

    const x1 = point[0] * cosYaw - point[1] * sinYaw;
    const y1 = point[0] * sinYaw + point[1] * cosYaw;
    const z1 = point[2];

    return [x1, y1 * cosPitch - z1 * sinPitch, y1 * sinPitch + z1 * cosPitch];
  }

  function projectPoint(point, scale, width, height) {
    const rotated = rotatePoint(point);
    return {
      x: width / 2 + state.panX + rotated[0] * scale,
      y: height / 2 + state.panY - rotated[1] * scale,
      depth: rotated[2],
    };
  }

  function getDatasetScale() {
    const datasets = [state.preview, state.ground, state.filtered, state.selected, state.voxels].filter(Boolean);
    if (datasets.length === 0) {
      return 1;
    }

    let maxExtent = 1;
    for (const dataset of datasets) {
      const extent = dataset.bbox.max.map((value, index) => value - dataset.bbox.min[index]);
      maxExtent = Math.max(maxExtent, ...extent);
    }
    const width = canvas.clientWidth || viewerStage.clientWidth || 1;
    const height = canvas.clientHeight || viewerStage.clientHeight || 1;
    return (0.42 * Math.min(width, height) * state.zoom) / maxExtent;
  }

  function drawDataset(dataset, fallbackColor, pointSize) {
    if (!dataset) {
      return [];
    }

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = getDatasetScale();
    const projected = [];
    const positions = dataset.positions;
    const colors = dataset.colors;

    for (let index = 0; index < positions.length; index += 3) {
      const point = [positions[index], positions[index + 1], positions[index + 2]];
      const projection = projectPoint(point, scale, width, height);
      projected.push({
        screenX: projection.x,
        screenY: projection.y,
        depth: projection.depth,
        world: point,
        color: [colors[index], colors[index + 1], colors[index + 2]],
      });
    }

    projected.sort((a, b) => a.depth - b.depth);
    for (const point of projected) {
      const color = point.color || fallbackColor;
      ctx.fillStyle = `rgb(${Math.round(color[0] * 255)}, ${Math.round(color[1] * 255)}, ${Math.round(color[2] * 255)})`;
      ctx.fillRect(point.screenX, point.screenY, pointSize, pointSize);
    }
    return projected;
  }

  function drawBoundingBox() {
    if (!state.bbox) {
      return;
    }

    const { min, max } = state.bbox;
    const corners = [
      [min[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], max[1], min[2]], [min[0], max[1], min[2]],
      [min[0], min[1], max[2]], [max[0], min[1], max[2]], [max[0], max[1], max[2]], [min[0], max[1], max[2]],
    ];
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = getDatasetScale();
    const projected = corners.map((corner) => projectPoint(corner, scale, width, height));

    ctx.strokeStyle = '#ffbf4d';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (const [start, end] of edges) {
      ctx.moveTo(projected[start].x, projected[start].y);
      ctx.lineTo(projected[end].x, projected[end].y);
    }
    ctx.stroke();
  }

  function drawPickedMarkers() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = getDatasetScale();
    ctx.fillStyle = '#ff6b35';
    for (const point of state.pickedPoints) {
      const projection = projectPoint(point, scale, width, height);
      ctx.beginPath();
      ctx.arc(projection.x, projection.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawSelectionBox() {
    if (!state.selectionBox) {
      return;
    }

    const { x, y, width, height } = state.selectionBox;
    ctx.save();
    ctx.fillStyle = 'rgba(184, 92, 56, 0.16)';
    ctx.strokeStyle = '#ffb38a';
    ctx.lineWidth = 1.5;
    ctx.fillRect(x, y, width, height);
    ctx.strokeRect(x, y, width, height);
    ctx.restore();
  }

  function draw() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#10131a';
    ctx.fillRect(0, 0, width, height);

    state.projectedPreview = drawDataset(state.preview, [0.85, 0.85, 0.85], 2);
    drawDataset(state.ground, [0.1, 0.75, 0.2], 2);
    drawDataset(state.filtered, [0.9, 0.2, 0.2], 2.5);
    drawDataset(state.selected, [0.15, 0.4, 0.95], 3);
    drawDataset(state.voxels, [0.12, 0.86, 1.0], 4);
    drawBoundingBox();
    drawPickedMarkers();
    drawSelectionBox();
  }

  function formatCount(value) {
    return Number(value).toLocaleString();
  }

  function setInteractionMode(mode) {
    state.interactionMode = mode;
    state.selectionBox = null;
    updatePickUI();
    draw();
  }

  function convexHull2D(points) {
    const sorted = [...points].sort((left, right) => (left[0] - right[0]) || (left[1] - right[1]) || (left[2] - right[2]));
    if (sorted.length <= 1) {
      return sorted;
    }

    const cross = (origin, a, b) => ((a[0] - origin[0]) * (b[1] - origin[1])) - ((a[1] - origin[1]) * (b[0] - origin[0]));
    const lower = [];
    for (const point of sorted) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
        lower.pop();
      }
      lower.push(point);
    }

    const upper = [];
    for (let index = sorted.length - 1; index >= 0; index -= 1) {
      const point = sorted[index];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
        upper.pop();
      }
      upper.push(point);
    }

    lower.pop();
    upper.pop();
    return lower.concat(upper);
  }

  function computeBounds(points) {
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const zs = points.map((point) => point[2]);
    return {
      min: [Math.min(...xs), Math.min(...ys), Math.min(...zs)],
      max: [Math.max(...xs), Math.max(...ys), Math.max(...zs)],
    };
  }

  async function loadPreview() {
    if (!state.currentFile) {
      setStatus('No point cloud file is available.', true);
      return;
    }

    setStatus(`Loading preview for ${state.currentFile}...`);
    state.preview = null;
    state.ground = null;
    state.filtered = null;
    state.selected = null;
    state.voxels = null;
    state.bbox = null;
    state.pickedPoints = [];
    state.selectionBounds = null;
    state.selectionKind = 'polygon';
    state.selectionBox = null;
    updatePickUI();
    updateResults();
    draw();

    try {
      const response = await fetch(`/api/preview?input_path=${encodeURIComponent(state.currentFile)}&voxel_size=${Number(inputs.previewVoxel.value)}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || 'Failed to load preview.');
      }
      state.preview = payload;
      updatePickUI();
      setStatus(`Preview opened: ${formatCount(payload.point_count)} points from ${state.currentFile}.`);
      draw();
    } catch (error) {
      setStatus(error.message || 'Failed to load preview.', true);
    }
  }

  function pickNearestPoint(clientX, clientY) {
    if (state.interactionMode !== 'point' || !state.preview || state.projectedPreview.length === 0) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let best = null;
    let bestDistance = 14;

    for (const point of state.projectedPreview) {
      const dx = point.screenX - x;
      const dy = point.screenY - y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = point;
      }
    }

    if (best) {
      state.selectionKind = 'polygon';
      state.selectionBounds = null;
      state.pickedPoints.push(best.world);
      updatePickUI();
      draw();
      setStatus(`Picked ${state.pickedPoints.length} point${state.pickedPoints.length === 1 ? "" : "s"} for the ROI footprint.`);
    }
  }

  function applyBoxSelection(selectionBox) {
    if (!state.preview || state.projectedPreview.length === 0) {
      return;
    }

    const left = Math.min(selectionBox.x, selectionBox.x + selectionBox.width);
    const right = Math.max(selectionBox.x, selectionBox.x + selectionBox.width);
    const top = Math.min(selectionBox.y, selectionBox.y + selectionBox.height);
    const bottom = Math.max(selectionBox.y, selectionBox.y + selectionBox.height);

    if ((right - left) < 8 || (bottom - top) < 8) {
      setStatus('Draw a larger box to capture the sand pile footprint.', true);
      return;
    }

    const selectedPoints = state.projectedPreview
      .filter((point) => point.screenX >= left && point.screenX <= right && point.screenY >= top && point.screenY <= bottom)
      .map((point) => point.world);

    if (selectedPoints.length < 3) {
      setStatus('The selection box did not capture enough preview points. Try a wider box.', true);
      return;
    }

    state.selectionKind = 'box';
    state.selectionBounds = computeBounds(selectedPoints);
    state.pickedPoints = convexHull2D(selectedPoints);
    updatePickUI();
    draw();
    setStatus(`Box selection captured ${formatCount(selectedPoints.length)} preview points. All cloud points inside that 3D region will be used.`);
  }

  async function analyze() {
    if (state.pickedPoints.length < 3) {
      setStatus('Pick at least 3 points around the sand pile footprint before analysis.', true);
      return;
    }

    setStatus('Running robust segmentation and volume estimation...');
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: state.currentFile,
          picked_points: state.pickedPoints,
          selection_mode: state.selectionKind,
          selection_bounds: state.selectionKind === 'box' ? state.selectionBounds : null,
          downsample_voxel: Number(inputs.downsampleVoxel.value),
          volume_voxel: Number(inputs.volumeVoxel.value),
          dbscan_eps: Number(inputs.dbscanEps.value),
          dbscan_min_points: Number(inputs.dbscanMinPoints.value),
          min_cluster_size: Number(inputs.minClusterSize.value),
          plane_threshold: Number(inputs.planeThreshold.value),
          height_threshold: Number(inputs.heightThreshold.value),
          roi_padding_xy: Number(inputs.roiPaddingXY.value),
          roi_padding_z: Number(inputs.roiPaddingZ.value),
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || 'Analysis failed.');
      }

      state.ground = payload.ground_cloud;
      state.filtered = payload.filtered_cloud;
      state.selected = payload.selected_cloud;
      state.voxels = payload.voxel_cloud;
      state.bbox = payload.bbox;
      updateResults([
        ['ROI points', formatCount(payload.roi_points)],
        ['Downsampled points', formatCount(payload.downsampled_points)],
        ['Denoised points', formatCount(payload.denoised_points)],
        ['Ground points', formatCount(payload.ground_points)],
        ['Filtered object points', formatCount(payload.object_points)],
        ['Points removed by clustering', formatCount(payload.points_removed_by_clustering)],
        ['Clusters detected', String(payload.cluster_count)],
        ['Cluster sizes', payload.cluster_sizes.join(', ') || 'none'],
        ['Selection strategy', payload.selected_strategy],
        ['Selected cluster', payload.selected_cluster_index < 0 ? 'ROI-driven selection' : String(payload.selected_cluster_index)],
        ['Selected points', formatCount(payload.selected_points)],
        ['Mean point spacing', `${payload.mean_point_spacing.toFixed(4)} m`],
        ['Ground fit RMSE', `${payload.ground_rmse.toFixed(4)} m`],
        ['ROI completeness', `${(payload.roi_completeness * 100).toFixed(1)}%`],
        ['Adaptive voxel size', `${payload.adaptive_voxel_size.toFixed(4)} m`],
        ['Occupied voxels', formatCount(payload.voxel_count)],
        ['Empty voxel ratio', `${payload.empty_voxel_percent.toFixed(2)}%`],
        ['BBox volume (comparison)', `${payload.bbox_volume_m3.toFixed(4)} m³`],
        ['Binary voxel volume', `${payload.binary_voxel_volume_m3.toFixed(4)} m³`],
        ['Weighted voxel volume', `${payload.weighted_voxel_volume_m3.toFixed(4)} m³`],
        ['Height map volume', `${payload.height_map_volume_m3.toFixed(4)} m³`],
        ['Mesh volume', `${payload.mesh_volume_m3.toFixed(4)} m³`],
        ['Final volume', `${payload.final_volume_m3.toFixed(4)} m³`],
        ['Confidence', payload.confidence],
        ['Estimated error', `${payload.error_estimate_percent.toFixed(2)}%`],
      ]);
      const strategyLabel = payload.selected_strategy === 'box_full_roi'
        ? 'Box ROI selection kept all filtered points inside the selected region.'
        : payload.selected_strategy === 'merged_roi_clusters'
          ? 'ROI-driven cluster merge selected the pile region.'
          : 'ROI-driven selection completed.';
      setStatus(`${strategyLabel} Final estimated sand volume: ${payload.final_volume_m3.toFixed(4)} m³. Method: ${payload.method_used}. Confidence: ${payload.confidence}.`);
      draw();
    } catch (error) {
      setStatus(error.message || 'Analysis failed.', true);
    }
  }

  canvas.addEventListener('mousedown', (event) => {
    state.dragging = true;
    state.dragButton = event.button;
    state.dragStartX = event.clientX;
    state.dragStartY = event.clientY;
    state.lastX = event.clientX;
    state.lastY = event.clientY;

    if (event.button === 0 && state.interactionMode === 'box') {
      const rect = canvas.getBoundingClientRect();
      state.selectionBox = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
        width: 0,
        height: 0,
      };
      updateInteractionUI();
      draw();
    }
  });

  window.addEventListener('mouseup', (event) => {
    if (!state.dragging) {
      return;
    }

    const moved = Math.abs(event.clientX - state.dragStartX) + Math.abs(event.clientY - state.dragStartY) > 4;
    const completedSelection = state.selectionBox ? { ...state.selectionBox } : null;

    state.dragging = false;
    state.selectionBox = null;
    updateInteractionUI();

    if (event.button === 0 && state.interactionMode === 'point' && !moved) {
      pickNearestPoint(event.clientX, event.clientY);
      return;
    }

    if (event.button === 0 && state.interactionMode === 'box' && completedSelection) {
      applyBoxSelection(completedSelection);
      return;
    }

    draw();
  });

  window.addEventListener('mousemove', (event) => {
    if (!state.dragging) {
      return;
    }

    const dx = event.clientX - state.lastX;
    const dy = event.clientY - state.lastY;
    state.lastX = event.clientX;
    state.lastY = event.clientY;

    if (state.dragButton === 0 && state.interactionMode === 'navigate') {
      state.yaw += dx * 0.01;
      state.pitch += dy * 0.01;
      state.pitch = Math.max(-1.4, Math.min(1.4, state.pitch));
    } else if (state.dragButton === 0 && state.interactionMode === 'box' && state.selectionBox) {
      const rect = canvas.getBoundingClientRect();
      state.selectionBox.width = event.clientX - rect.left - state.selectionBox.x;
      state.selectionBox.height = event.clientY - rect.top - state.selectionBox.y;
    } else if (state.dragButton === 2) {
      state.panX += dx;
      state.panY += dy;
    }

    updateInteractionUI();
    draw();
  });

  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    event.stopPropagation();
    state.zoom *= event.deltaY > 0 ? 0.92 : 1.08;
    state.zoom = Math.max(0.2, Math.min(20, state.zoom));
    draw();
  }, { passive: false });

  canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  window.addEventListener('resize', resizeCanvas);

  viewerStage.addEventListener('wheel', (event) => {
    if (event.target === canvas) {
      return;
    }
    event.stopPropagation();
  }, { passive: true });

  if (navigateModeButton) {
    navigateModeButton.addEventListener('click', () => {
      setInteractionMode('navigate');
      setStatus('Navigation mode enabled. Orbit with left drag, pan with right drag.');
    });
  }

  if (pointPickModeButton) {
    pointPickModeButton.addEventListener('click', () => {
      setInteractionMode('point');
      setStatus('Point pick mode enabled. Click around the sand pile footprint.');
    });
  }

  if (boxPickModeButton) {
    boxPickModeButton.addEventListener('click', () => {
      setInteractionMode('box');
      setStatus('Box selection enabled. Drag a rectangle over the sand pile and all points in that region will be included.');
    });
  }

  clearPicksButton.addEventListener('click', () => {
    state.pickedPoints = [];
    state.selectionBounds = null;
    state.selectionKind = 'polygon';
    state.selectionBox = null;
    updatePickUI();
    draw();
    setStatus('Picked points cleared.');
  });

  analyzeButton.addEventListener('click', analyze);
  loadPreviewButton.addEventListener('click', loadPreview);
  fileSelect.addEventListener('change', () => {
    state.currentFile = fileSelect.value;
    setStatus(`Selected dataset: ${state.currentFile}. Click Load Preview.`);
  });

  updatePickUI();
  resizeCanvas();
  if (state.currentFile) {
    fileSelect.value = state.currentFile;
    loadPreview();
  } else {
    setStatus('No .ply or .pcd files found in the workspace root.', true);
  }
})();









