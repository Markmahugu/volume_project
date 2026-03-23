(() => {
  const canvas = document.getElementById('viewerCanvas');
  const viewerStage = document.querySelector('.viewer-stage');
  const fileSelect = document.getElementById('fileSelect');
  const refreshFilesButton = document.getElementById('refreshFilesButton');
  const loadPreviewButton = document.getElementById('loadPreviewButton');
  const navigateModeButton = document.getElementById('navigateModeButton');
  const pointPickModeButton = document.getElementById('pointPickModeButton');
  const boxPickModeButton = document.getElementById('boxPickModeButton');
  const cuboidModeButton = document.getElementById('cuboidModeButton');
  const clearPicksButton = document.getElementById('clearPicksButton');
  const analyzeButton = document.getElementById('analyzeButton');
  const selectionModeLabel = document.getElementById('selectionModeLabel');
  const pickCount = document.getElementById('pickCount');
  const pickList = document.getElementById('pickList');
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  const cuboidMeta = document.getElementById('cuboidMeta');
  const snapToGroundToggle = document.getElementById('snapToGroundToggle');
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
    selectionCuboid: null,
    cuboidStats: null,
    cuboidDrag: null,
    cuboidStatsTimer: null,
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

  const CUBOID_MIN_SIZE = 0.1;
  const CUBOID_ROTATE_SENSITIVITY = 0.012;

  function setStatus(message, isError = false) {
    status.textContent = message;
    status.style.color = isError ? '#b42318' : '#6e6254';
  }

  function updateDefinitionList(target, entries = []) {
    if (!target) return;
    target.innerHTML = '';
    for (const [label, value] of entries) {
      const dt = document.createElement('dt');
      dt.textContent = label;
      const dd = document.createElement('dd');
      dd.textContent = value;
      target.append(dt, dd);
    }
  }

  function updateResults(entries = []) {
    updateDefinitionList(results, entries);
  }

  function formatCount(value) {
    return Number(value).toLocaleString();
  }

  function pixelScale() {
    return window.devicePixelRatio || 1;
  }

  function updateFileOptions(files) {
    fileSelect.innerHTML = '';
    if (!files.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'No point cloud files found';
      fileSelect.appendChild(option);
      loadPreviewButton.disabled = true;
      state.currentFile = '';
      return;
    }

    for (const file of files) {
      const option = document.createElement('option');
      option.value = file;
      option.textContent = file;
      fileSelect.appendChild(option);
    }

    if (state.currentFile && files.includes(state.currentFile)) {
      fileSelect.value = state.currentFile;
    } else {
      state.currentFile = files[0];
      fileSelect.value = state.currentFile;
    }
    loadPreviewButton.disabled = false;
  }

  async function refreshFiles() {
    setStatus('Refreshing dataset list...');
    try {
      const response = await fetch('/api/files');
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Failed to refresh file list.');
      const files = Array.isArray(payload.files) ? payload.files : [];
      updateFileOptions(files);
      setStatus(files.length ? `Dataset list refreshed. Selected: ${state.currentFile}.` : 'No .ply or .pcd files found.', !files.length);
    } catch (error) {
      setStatus(error.message || 'Failed to refresh file list.', true);
    }
  }

  function updateInteractionUI() {
    const isNavigate = state.interactionMode === 'navigate';
    const isPoint = state.interactionMode === 'point';
    const isBox = state.interactionMode === 'box';
    const isCuboid = state.interactionMode === 'cuboid';

    if (navigateModeButton) navigateModeButton.classList.toggle('is-active', isNavigate);
    if (pointPickModeButton) pointPickModeButton.classList.toggle('is-active', isPoint);
    if (boxPickModeButton) boxPickModeButton.classList.toggle('is-active', isBox);
    if (cuboidModeButton) cuboidModeButton.classList.toggle('is-active', isCuboid);

    viewerStage.classList.toggle('mode-point', isPoint);
    viewerStage.classList.toggle('mode-box', isBox);
    viewerStage.classList.toggle('mode-cuboid', isCuboid);
    viewerStage.classList.toggle('is-box-dragging', Boolean(state.selectionBox));

    if (!selectionModeLabel) return;
    selectionModeLabel.textContent = isNavigate
      ? 'Navigation mode active.'
      : isPoint
        ? 'Point mode active. Click around the object footprint.'
        : isBox
          ? 'Box mode active. Drag a 2D box; all cloud points inside that region will be included.'
          : 'Cuboid mode active. Click to place a 3D ROI, then drag center, faces, corners, top, or rotate handle.';
  }

  function updateCuboidMeta() {
    if (!cuboidMeta) return;
    if (!state.selectionCuboid) {
      updateDefinitionList(cuboidMeta, []);
      return;
    }

    const c = state.selectionCuboid.center;
    const d = state.selectionCuboid.dimensions;
    const stats = state.cuboidStats;
    updateDefinitionList(cuboidMeta, [
      ['Center', c.map((value) => value.toFixed(2)).join(', ')],
      ['Dimensions', d.map((value) => value.toFixed(2)).join(' x ')],
      ['Rotation', `${(state.selectionCuboid.yaw * 180 / Math.PI).toFixed(1)} deg`],
      ['Cuboid Volume', `${(d[0] * d[1] * d[2]).toFixed(3)} m³`],
      ['Selected Points', stats ? formatCount(stats.selected_points) : '...'],
      ['Excluded Points', stats ? formatCount(stats.excluded_points) : '...'],
      ['Ground Z', stats && Number.isFinite(stats.ground_z) ? stats.ground_z.toFixed(2) : 'n/a'],
    ]);
  }

  function updatePickUI() {
    if (state.selectionKind === 'cuboid' && state.selectionCuboid) {
      pickCount.textContent = 'Cuboid ROI active';
      pickList.innerHTML = '';
    } else {
      pickCount.textContent = `Picked points: ${state.pickedPoints.length}`;
      pickList.innerHTML = '';
      for (let index = 0; index < state.pickedPoints.length; index += 1) {
        const point = state.pickedPoints[index];
        const item = document.createElement('li');
        item.textContent = `${index + 1}: ${point.map((value) => value.toFixed(2)).join(', ')}`;
        pickList.appendChild(item);
      }
    }

    analyzeButton.disabled = !state.preview || !((state.selectionKind === 'cuboid' && state.selectionCuboid) || state.pickedPoints.length >= 3);
    updateInteractionUI();
    updateCuboidMeta();
  }

  function resizeCanvas() {
    const width = Math.max(1, viewerStage.clientWidth);
    const height = Math.max(1, viewerStage.clientHeight);
    canvas.width = Math.floor(width * pixelScale());
    canvas.height = Math.floor(height * pixelScale());
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(pixelScale(), 0, 0, pixelScale(), 0, 0);
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
    if (datasets.length === 0) return 1;

    let maxExtent = 1;
    for (const dataset of datasets) {
      if (!dataset?.bbox) continue;
      const extent = dataset.bbox.max.map((value, index) => value - dataset.bbox.min[index]);
      maxExtent = Math.max(maxExtent, ...extent);
    }
    if (state.selectionCuboid) maxExtent = Math.max(maxExtent, ...state.selectionCuboid.dimensions);

    const width = canvas.clientWidth || viewerStage.clientWidth || 1;
    const height = canvas.clientHeight || viewerStage.clientHeight || 1;
    return (0.42 * Math.min(width, height) * state.zoom) / maxExtent;
  }

  function drawDataset(dataset, fallbackColor, pointSize) {
    if (!dataset || !dataset.positions) return [];
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = getDatasetScale();
    const projected = [];
    const positions = dataset.positions;
    const colors = dataset.colors || [];

    for (let index = 0; index < positions.length; index += 3) {
      const point = [positions[index], positions[index + 1], positions[index + 2]];
      const projection = projectPoint(point, scale, width, height);
      projected.push({
        screenX: projection.x,
        screenY: projection.y,
        depth: projection.depth,
        world: point,
        color: colors.length ? [colors[index], colors[index + 1], colors[index + 2]] : fallbackColor,
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
    if (!state.bbox) return;
    const { min, max } = state.bbox;
    const corners = [
      [min[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], max[1], min[2]], [min[0], max[1], min[2]],
      [min[0], min[1], max[2]], [max[0], min[1], max[2]], [max[0], max[1], max[2]], [min[0], max[1], max[2]],
    ];
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = getDatasetScale();
    const projected = corners.map((corner) => projectPoint(corner, scale, width, height));

    ctx.save();
    ctx.strokeStyle = '#ffbf4d';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (const [start, end] of edges) {
      ctx.moveTo(projected[start].x, projected[start].y);
      ctx.lineTo(projected[end].x, projected[end].y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawPickedMarkers() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = getDatasetScale();
    ctx.save();
    ctx.fillStyle = '#ff6b35';
    for (const point of state.pickedPoints) {
      const projection = projectPoint(point, scale, width, height);
      ctx.beginPath();
      ctx.arc(projection.x, projection.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawSelectionBox() {
    if (!state.selectionBox) return;
    const { x, y, width, height } = state.selectionBox;
    ctx.save();
    ctx.fillStyle = 'rgba(184, 92, 56, 0.16)';
    ctx.strokeStyle = '#ffb38a';
    ctx.lineWidth = 1.5;
    ctx.fillRect(x, y, width, height);
    ctx.strokeRect(x, y, width, height);
    ctx.restore();
  }

  function rotateCuboidLocal(localPoint, yaw) {
    const cosYaw = Math.cos(yaw);
    const sinYaw = Math.sin(yaw);
    return [
      localPoint[0] * cosYaw - localPoint[1] * sinYaw,
      localPoint[0] * sinYaw + localPoint[1] * cosYaw,
      localPoint[2],
    ];
  }

  function cuboidLocalToWorld(localPoint, yaw) {
    return rotateCuboidLocal(localPoint, yaw);
  }

  function cuboidWorldToLocal(worldPoint, cuboid) {
    const dx = worldPoint[0] - cuboid.center[0];
    const dy = worldPoint[1] - cuboid.center[1];
    const dz = worldPoint[2] - cuboid.center[2];
    const cosYaw = Math.cos(cuboid.yaw);
    const sinYaw = Math.sin(cuboid.yaw);
    return [dx * cosYaw + dy * sinYaw, -dx * sinYaw + dy * cosYaw, dz];
  }

  function getCuboidCorners() {
    if (!state.selectionCuboid) return [];
    const { center, dimensions, yaw } = state.selectionCuboid;
    const [hx, hy, hz] = dimensions.map((value) => value / 2);
    const locals = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ];
    return locals.map((local) => {
      const rotated = rotateCuboidLocal(local, yaw);
      return [center[0] + rotated[0], center[1] + rotated[1], center[2] + rotated[2]];
    });
  }

  function getCuboidHandles() {
    if (!state.selectionCuboid) return [];
    const { center, dimensions, yaw } = state.selectionCuboid;
    const [hx, hy, hz] = dimensions.map((value) => value / 2);
    const cornerOffset = 0.22 * Math.max(dimensions[0], dimensions[1], dimensions[2]);
    const specs = [
      { kind: 'center', local: [0, 0, 0], radius: 6 },
      { kind: 'top', local: [0, 0, hz], radius: 5 },
      { kind: 'rotate', local: [0, hy + Math.max(0.35, cornerOffset), hz], radius: 5 },
      { kind: 'faceX+', local: [hx, 0, 0], radius: 5, axis: [1, 0, 0] },
      { kind: 'faceX-', local: [-hx, 0, 0], radius: 5, axis: [-1, 0, 0] },
      { kind: 'faceY+', local: [0, hy, 0], radius: 5, axis: [0, 1, 0] },
      { kind: 'faceY-', local: [0, -hy, 0], radius: 5, axis: [0, -1, 0] },
      { kind: 'corner++', local: [hx, hy, hz], radius: 5, axis: [1, 1, 1] },
      { kind: 'corner+-', local: [hx, -hy, hz], radius: 5, axis: [1, -1, 1] },
      { kind: 'corner-+', local: [-hx, hy, hz], radius: 5, axis: [-1, 1, 1] },
      { kind: 'corner--', local: [-hx, -hy, hz], radius: 5, axis: [-1, -1, 1] },
    ];

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = getDatasetScale();
    return specs.map((spec) => {
      const rotated = rotateCuboidLocal(spec.local, yaw);
      const world = [center[0] + rotated[0], center[1] + rotated[1], center[2] + rotated[2]];
      const projected = projectPoint(world, scale, width, height);
      return { ...spec, world, screenX: projected.x, screenY: projected.y };
    });
  }

  function drawCuboid() {
    if (!state.selectionCuboid) return;
    const corners = getCuboidCorners();
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = getDatasetScale();
    const projected = corners.map((corner) => projectPoint(corner, scale, width, height));
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    const faces = [[4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];

    ctx.save();
    ctx.fillStyle = 'rgba(61, 179, 255, 0.10)';
    for (const face of faces) {
      ctx.beginPath();
      ctx.moveTo(projected[face[0]].x, projected[face[0]].y);
      for (let index = 1; index < face.length; index += 1) {
        ctx.lineTo(projected[face[index]].x, projected[face[index]].y);
      }
      ctx.closePath();
      ctx.fill();
    }

    ctx.strokeStyle = '#3db3ff';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (const [start, end] of edges) {
      ctx.moveTo(projected[start].x, projected[start].y);
      ctx.lineTo(projected[end].x, projected[end].y);
    }
    ctx.stroke();

    for (const handle of getCuboidHandles()) {
      ctx.beginPath();
      if (handle.kind === 'rotate') {
        ctx.fillStyle = '#ffd166';
      } else if (handle.kind === 'top') {
        ctx.fillStyle = '#58d68d';
      } else if (handle.kind.startsWith('corner')) {
        ctx.fillStyle = '#f8f9fa';
      } else if (handle.kind === 'center') {
        ctx.fillStyle = '#3db3ff';
      } else {
        ctx.fillStyle = '#ffffff';
      }
      ctx.strokeStyle = '#10131a';
      ctx.lineWidth = 1.2;
      ctx.arc(handle.screenX, handle.screenY, handle.radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
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
    drawCuboid();
  }

  function setInteractionMode(mode) {
    state.interactionMode = mode;
    state.selectionBox = null;
    updatePickUI();
    draw();
  }

  function convexHull2D(points) {
    const sorted = [...points].sort((left, right) => (left[0] - right[0]) || (left[1] - right[1]) || (left[2] - right[2]));
    if (sorted.length <= 1) return sorted;

    const cross = (origin, a, b) => ((a[0] - origin[0]) * (b[1] - origin[1])) - ((a[1] - origin[1]) * (b[0] - origin[0]));
    const lower = [];
    for (const point of sorted) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
      lower.push(point);
    }

    const upper = [];
    for (let index = sorted.length - 1; index >= 0; index -= 1) {
      const point = sorted[index];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
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
    return { min: [Math.min(...xs), Math.min(...ys), Math.min(...zs)], max: [Math.max(...xs), Math.max(...ys), Math.max(...zs)] };
  }

  function getPreviewPoints() {
    if (!state.preview?.positions) return [];
    const points = [];
    for (let index = 0; index < state.preview.positions.length; index += 3) {
      points.push([state.preview.positions[index], state.preview.positions[index + 1], state.preview.positions[index + 2]]);
    }
    return points;
  }

  function updateCuboidGroundSnap() {
    if (!state.selectionCuboid || !snapToGroundToggle || !state.preview) return;
    state.selectionCuboid.snap_to_ground = Boolean(snapToGroundToggle.checked);
    if (!state.selectionCuboid.snap_to_ground) return;

    const halfX = state.selectionCuboid.dimensions[0] / 2;
    const halfY = state.selectionCuboid.dimensions[1] / 2;
    const candidates = [];
    for (const point of getPreviewPoints()) {
      const local = cuboidWorldToLocal(point, state.selectionCuboid);
      if (Math.abs(local[0]) <= halfX + 0.2 && Math.abs(local[1]) <= halfY + 0.2) {
        candidates.push(point[2]);
      }
    }

    if (candidates.length) {
      candidates.sort((left, right) => left - right);
      const groundZ = candidates[Math.floor(candidates.length * 0.05)];
      state.selectionCuboid.ground_z = groundZ;
      state.selectionCuboid.center[2] = groundZ + (state.selectionCuboid.dimensions[2] / 2);
    }
  }

  function debounceCuboidStats() {
    if (state.cuboidStatsTimer) clearTimeout(state.cuboidStatsTimer);
    state.cuboidStatsTimer = setTimeout(requestCuboidStats, 120);
  }

  async function requestCuboidStats() {
    if (!state.currentFile || !state.selectionCuboid) return;
    try {
      const response = await fetch('/api/cuboid-stats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path: state.currentFile, selection_cuboid: state.selectionCuboid }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Failed to update cuboid stats.');
      state.selectionCuboid.center = payload.center;
      state.selectionCuboid.ground_z = payload.ground_z;
      state.cuboidStats = payload;
      updateCuboidMeta();
      draw();
    } catch (error) {
      setStatus(error.message || 'Failed to update cuboid stats.', true);
    }
  }

  function pickNearestPoint(clientX, clientY) {
    if (!state.preview || state.projectedPreview.length === 0) return null;
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
    return best;
  }

  function createCuboidAt(point) {
    const bbox = state.preview ? state.preview.bbox : { min: [0, 0, 0], max: [1, 1, 1] };
    const extent = bbox.max.map((value, index) => value - bbox.min[index]);
    const base = Math.max(0.6, 0.12 * Math.max(...extent));
    state.selectionKind = 'cuboid';
    state.selectionBounds = null;
    state.pickedPoints = [];
    state.selectionCuboid = {
      center: [point[0], point[1], point[2]],
      dimensions: [base, base, Math.max(base, 1.0)],
      yaw: 0,
      snap_to_ground: Boolean(snapToGroundToggle?.checked),
      ground_z: null,
    };
    updateCuboidGroundSnap();
    state.cuboidStats = null;
    debounceCuboidStats();
    updatePickUI();
  }
  function subtractVec2(left, right) {
    return [left[0] - right[0], left[1] - right[1]];
  }

  function lengthVec2(vector) {
    return Math.sqrt((vector[0] * vector[0]) + (vector[1] * vector[1]));
  }

  function dotVec2(left, right) {
    return (left[0] * right[0]) + (left[1] * right[1]);
  }

  function normalizeVec2(vector) {
    const length = lengthVec2(vector);
    if (length < 1e-6) return [0, 0];
    return [vector[0] / length, vector[1] / length];
  }

  function getScreenPointForWorld(worldPoint) {
    const projection = projectPoint(worldPoint, getDatasetScale(), canvas.clientWidth, canvas.clientHeight);
    return [projection.x, projection.y];
  }

  function projectWorldAxisToScreen(originWorld, axisWorld) {
    const originScreen = getScreenPointForWorld(originWorld);
    const axisEndScreen = getScreenPointForWorld([
      originWorld[0] + axisWorld[0],
      originWorld[1] + axisWorld[1],
      originWorld[2] + axisWorld[2],
    ]);
    return subtractVec2(axisEndScreen, originScreen);
  }

  function getCuboidWorldAxes(cuboid) {
    return {
      x: cuboidLocalToWorld([1, 0, 0], cuboid.yaw),
      y: cuboidLocalToWorld([0, 1, 0], cuboid.yaw),
      z: [0, 0, 1],
    };
  }

  function computeAxisDeltaFromMouse(originWorld, axisWorld, mouseDx, mouseDy) {
    const axisScreen = projectWorldAxisToScreen(originWorld, axisWorld);
    const axisScreenLength = lengthVec2(axisScreen);
    if (axisScreenLength < 1e-6) return 0;
    const axisScreenUnit = normalizeVec2(axisScreen);
    return dotVec2([mouseDx, mouseDy], axisScreenUnit) / axisScreenLength;
  }

  function getHandleDragConfig(handle) {
    if (!handle) return null;
    if (handle.kind === 'center' || handle.kind === 'top' || handle.kind === 'rotate') {
      return { kind: handle.kind };
    }
    if (handle.kind.startsWith('face')) {
      return { kind: 'face', axis: handle.axis };
    }
    if (handle.kind.startsWith('corner')) {
      return { kind: 'corner', axis: handle.axis };
    }
    return null;
  }

  function hitTestCuboidHandle(clientX, clientY) {
    if (!state.selectionCuboid) return null;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let best = null;
    let bestDistance = 12;

    for (const handle of getCuboidHandles()) {
      const dx = handle.screenX - x;
      const dy = handle.screenY - y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = handle;
      }
    }
    return best;
  }

  function clampDimension(value) {
    return Math.max(CUBOID_MIN_SIZE, value);
  }

  function moveCuboidByWorldDelta(worldDelta) {
    const cuboid = state.selectionCuboid;
    cuboid.center[0] += worldDelta[0];
    cuboid.center[1] += worldDelta[1];
    cuboid.center[2] += worldDelta[2] || 0;
  }

  function applyFaceResize(axis, mouseDx, mouseDy, modifiers) {
    const cuboid = state.selectionCuboid;
    const axes = getCuboidWorldAxes(cuboid);
    const axisWorld = axis[0] !== 0 ? axes.x : axes.y;
    const axisSign = axis[0] !== 0 ? Math.sign(axis[0]) : Math.sign(axis[1]);
    const axisIndex = axis[0] !== 0 ? 0 : 1;
    const delta = computeAxisDeltaFromMouse(cuboid.center, axisWorld, mouseDx, mouseDy) * axisSign;
    if (Math.abs(delta) < 1e-6) return;

    const oldDimension = cuboid.dimensions[axisIndex];
    const newDimension = clampDimension(oldDimension + delta);
    const effectiveDelta = newDimension - oldDimension;
    cuboid.dimensions[axisIndex] = newDimension;

    if (!modifiers.altKey) {
      const centerShift = cuboidLocalToWorld(
        axisIndex === 0 ? [axisSign * effectiveDelta / 2, 0, 0] : [0, axisSign * effectiveDelta / 2, 0],
        cuboid.yaw,
      );
      moveCuboidByWorldDelta([centerShift[0], centerShift[1], 0]);
    }

    if (modifiers.shiftKey) {
      const otherIndex = axisIndex === 0 ? 1 : 0;
      cuboid.dimensions[otherIndex] = clampDimension(cuboid.dimensions[otherIndex] + effectiveDelta);
    }
  }

  function applyCornerDimension(index, delta, scaleFromCenter, signOverride = 1) {
    const cuboid = state.selectionCuboid;
    if (Math.abs(delta) < 1e-6) return;

    const oldDimension = cuboid.dimensions[index];
    const newDimension = clampDimension(oldDimension + delta);
    const effectiveDelta = newDimension - oldDimension;
    cuboid.dimensions[index] = newDimension;
    if (scaleFromCenter) return;

    if (index === 2) {
      cuboid.center[2] += signOverride * effectiveDelta / 2;
      return;
    }

    const worldShift = cuboidLocalToWorld(index === 0 ? [signOverride * effectiveDelta / 2, 0, 0] : [0, signOverride * effectiveDelta / 2, 0], cuboid.yaw);
    moveCuboidByWorldDelta([worldShift[0], worldShift[1], 0]);
  }

  function applyCornerResize(axis, mouseDx, mouseDy, modifiers) {
    const cuboid = state.selectionCuboid;
    const axes = getCuboidWorldAxes(cuboid);
    const deltaX = computeAxisDeltaFromMouse(cuboid.center, axes.x, mouseDx, mouseDy) * Math.sign(axis[0]);
    const deltaY = computeAxisDeltaFromMouse(cuboid.center, axes.y, mouseDx, mouseDy) * Math.sign(axis[1]);
    const deltaZ = computeAxisDeltaFromMouse(cuboid.center, axes.z, mouseDx, mouseDy) * Math.sign(axis[2]);

    if (modifiers.shiftKey) {
      const magnitude = Math.max(Math.abs(deltaX), Math.abs(deltaY), Math.abs(deltaZ));
      const signed = Math.sign(deltaX || deltaY || deltaZ || 1) * magnitude;
      applyCornerDimension(0, signed * Math.sign(axis[0]), modifiers.altKey, Math.sign(axis[0]));
      applyCornerDimension(1, signed * Math.sign(axis[1]), modifiers.altKey, Math.sign(axis[1]));
      applyCornerDimension(2, signed * Math.sign(axis[2]), modifiers.altKey, Math.sign(axis[2]));
      return;
    }

    applyCornerDimension(0, deltaX, modifiers.altKey, Math.sign(axis[0]));
    applyCornerDimension(1, deltaY, modifiers.altKey, Math.sign(axis[1]));
    applyCornerDimension(2, deltaZ, modifiers.altKey, Math.sign(axis[2]));
  }

  function applyCuboidDrag(mouseDx, mouseDy, event) {
    if (!state.selectionCuboid || !state.cuboidDrag) return;
    const cuboid = state.selectionCuboid;

    switch (state.cuboidDrag.kind) {
      case 'center': {
        if (!cuboid.snap_to_ground && event.shiftKey) {
          const deltaZ = computeAxisDeltaFromMouse(cuboid.center, [0, 0, 1], mouseDx, mouseDy);
          cuboid.center[2] += deltaZ;
        } else {
          const axes = getCuboidWorldAxes(cuboid);
          const moveX = computeAxisDeltaFromMouse(cuboid.center, axes.x, mouseDx, mouseDy);
          const moveY = computeAxisDeltaFromMouse(cuboid.center, axes.y, mouseDx, mouseDy);
          const worldX = cuboidLocalToWorld([moveX, 0, 0], cuboid.yaw);
          const worldY = cuboidLocalToWorld([0, moveY, 0], cuboid.yaw);
          moveCuboidByWorldDelta([worldX[0] + worldY[0], worldX[1] + worldY[1], 0]);
        }
        break;
      }
      case 'top': {
        const deltaHeight = computeAxisDeltaFromMouse(cuboid.center, [0, 0, 1], mouseDx, mouseDy);
        const oldHeight = cuboid.dimensions[2];
        cuboid.dimensions[2] = clampDimension(oldHeight + deltaHeight);
        const effectiveDelta = cuboid.dimensions[2] - oldHeight;
        if (!cuboid.snap_to_ground) {
          cuboid.center[2] += effectiveDelta / 2;
        }
        break;
      }
      case 'rotate': {
        cuboid.yaw += mouseDx * CUBOID_ROTATE_SENSITIVITY;
        break;
      }
      case 'face': {
        applyFaceResize(state.cuboidDrag.axis, mouseDx, mouseDy, event);
        break;
      }
      case 'corner': {
        applyCornerResize(state.cuboidDrag.axis, mouseDx, mouseDy, event);
        break;
      }
      default:
        break;
    }

    if (cuboid.snap_to_ground) updateCuboidGroundSnap();
    debounceCuboidStats();
    updateCuboidMeta();
    draw();
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
    state.selectionCuboid = null;
    state.cuboidStats = null;
    state.selectionKind = 'polygon';
    state.selectionBox = null;
    updatePickUI();
    updateResults();
    draw();

    try {
      const response = await fetch(`/api/preview?input_path=${encodeURIComponent(state.currentFile)}&voxel_size=${Number(inputs.previewVoxel.value)}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Failed to load preview.');
      state.preview = payload;
      updatePickUI();
      setStatus(`Preview opened: ${formatCount(payload.point_count)} points from ${state.currentFile}.`);
      draw();
    } catch (error) {
      setStatus(error.message || 'Failed to load preview.', true);
    }
  }

  function applyBoxSelection(selectionBox) {
    if (!state.preview || state.projectedPreview.length === 0) return;
    const left = Math.min(selectionBox.x, selectionBox.x + selectionBox.width);
    const right = Math.max(selectionBox.x, selectionBox.x + selectionBox.width);
    const top = Math.min(selectionBox.y, selectionBox.y + selectionBox.height);
    const bottom = Math.max(selectionBox.y, selectionBox.y + selectionBox.height);
    if ((right - left) < 8 || (bottom - top) < 8) {
      setStatus('Draw a larger box to capture the object footprint.', true);
      return;
    }

    const selectedPoints = state.projectedPreview.filter((point) => point.screenX >= left && point.screenX <= right && point.screenY >= top && point.screenY <= bottom).map((point) => point.world);
    if (selectedPoints.length < 3) {
      setStatus('The selection box did not capture enough preview points. Try a wider box.', true);
      return;
    }

    state.selectionKind = 'box';
    state.selectionCuboid = null;
    state.cuboidStats = null;
    state.selectionBounds = computeBounds(selectedPoints);
    state.pickedPoints = convexHull2D(selectedPoints);
    updatePickUI();
    draw();
    setStatus(`Box selection captured ${formatCount(selectedPoints.length)} preview points. All cloud points inside that 3D region will be used.`);
  }
  async function analyze() {
    const hasCuboid = state.selectionKind === 'cuboid' && state.selectionCuboid;
    if (!hasCuboid && state.pickedPoints.length < 3) {
      setStatus('Create a cuboid or pick at least 3 points before analysis.', true);
      return;
    }

    setStatus('Running robust segmentation and volume estimation...');
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: state.currentFile,
          picked_points: hasCuboid ? [] : state.pickedPoints,
          selection_mode: hasCuboid ? 'cuboid' : state.selectionKind,
          selection_bounds: state.selectionKind === 'box' ? state.selectionBounds : null,
          selection_cuboid: hasCuboid ? state.selectionCuboid : null,
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
      if (!response.ok) throw new Error(payload.detail || 'Analysis failed.');

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
        ['Selection strategy', payload.selected_strategy],
        ['Selected points', formatCount(payload.selected_points)],
        ['Mean point spacing', `${payload.mean_point_spacing.toFixed(4)} m`],
        ['Ground fit RMSE', `${payload.ground_rmse.toFixed(4)} m`],
        ['Adaptive voxel size', `${payload.adaptive_voxel_size.toFixed(4)} m`],
        ['Occupied voxels', formatCount(payload.voxel_count)],
        ['Empty voxel ratio', `${payload.empty_voxel_percent.toFixed(2)}%`],
        ['BBox volume (comparison)', `${payload.bbox_volume_m3.toFixed(4)} m³`],
        ['Binary voxel volume', `${payload.binary_voxel_volume_m3.toFixed(4)} m³`],
        ['Weighted voxel volume', `${payload.weighted_voxel_volume_m3.toFixed(4)} m³`],
        ['Height map volume', `${payload.height_map_volume_m3.toFixed(4)} m³`],
        ['Mesh volume', `${payload.mesh_volume_m3.toFixed(4)} m³`],
        ['Final volume', `${payload.final_volume_m3.toFixed(4)} m³`],
        ['Method used', payload.method_used],
        ['Confidence', payload.confidence],
        ['Warnings', payload.warnings && payload.warnings.length ? payload.warnings.join(' | ') : 'none'],
        ['Estimated error', `${payload.error_estimate_percent.toFixed(2)}%`],
      ]);

      const strategyLabel = payload.selected_strategy === 'cuboid_full_roi'
        ? 'Cuboid ROI selected points strictly inside the transformed 3D box.'
        : payload.selected_strategy === 'box_full_roi'
          ? 'Box ROI selection kept all filtered points inside the selected region.'
          : payload.selected_strategy === 'merged_roi_clusters'
            ? 'ROI-driven cluster merge selected the object region.'
            : 'ROI-driven selection completed.';

      setStatus(`${strategyLabel} Estimated object volume: ${payload.final_volume_m3.toFixed(4)} m³. Method: ${payload.method_used}. Confidence: ${payload.confidence}.${payload.warnings && payload.warnings.length ? ` Warning: ${payload.warnings[0]}` : ''}`);
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

    if (event.button !== 0) return;

    if (state.interactionMode === 'box') {
      const rect = canvas.getBoundingClientRect();
      state.selectionBox = { x: event.clientX - rect.left, y: event.clientY - rect.top, width: 0, height: 0 };
      updateInteractionUI();
      draw();
      return;
    }

    if (state.interactionMode === 'cuboid') {
      const handle = hitTestCuboidHandle(event.clientX, event.clientY);
      if (handle) {
        state.cuboidDrag = getHandleDragConfig(handle);
        return;
      }
      const nearest = pickNearestPoint(event.clientX, event.clientY);
      if (nearest) {
        createCuboidAt(nearest.world);
        setStatus('Cuboid ROI created. Drag handles to move, resize, or rotate it. Hold Shift on the center handle to move it vertically. Hold Alt while resizing to scale from the center.');
      }
    }
  });

  window.addEventListener('mouseup', (event) => {
    if (!state.dragging) return;
    const moved = Math.abs(event.clientX - state.dragStartX) + Math.abs(event.clientY - state.dragStartY) > 4;
    const completedSelection = state.selectionBox ? { ...state.selectionBox } : null;
    state.dragging = false;
    state.selectionBox = null;
    state.cuboidDrag = null;
    updateInteractionUI();

    if (event.button === 0 && state.interactionMode === 'point' && !moved) {
      const best = pickNearestPoint(event.clientX, event.clientY);
      if (best) {
        state.selectionKind = 'polygon';
        state.selectionBounds = null;
        state.selectionCuboid = null;
        state.cuboidStats = null;
        state.pickedPoints.push(best.world);
        updatePickUI();
        draw();
        setStatus(`Picked ${state.pickedPoints.length} point${state.pickedPoints.length === 1 ? '' : 's'} for the ROI footprint.`);
      }
      return;
    }

    if (event.button === 0 && state.interactionMode === 'box' && completedSelection) {
      applyBoxSelection(completedSelection);
      return;
    }

    if (state.selectionKind === 'cuboid') debounceCuboidStats();
    draw();
  });

  window.addEventListener('mousemove', (event) => {
    if (!state.dragging) return;
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
    } else if (state.dragButton === 0 && state.interactionMode === 'cuboid' && state.cuboidDrag) {
      applyCuboidDrag(dx, dy, event);
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
    if (event.target === canvas) return;
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
      setStatus('Point pick mode enabled. Click around the object footprint.');
    });
  }
  if (boxPickModeButton) {
    boxPickModeButton.addEventListener('click', () => {
      setInteractionMode('box');
      setStatus('Box selection enabled. Drag a rectangle over the target object and all points in that region will be included.');
    });
  }
  if (cuboidModeButton) {
    cuboidModeButton.addEventListener('click', () => {
      setInteractionMode('cuboid');
      setStatus('Cuboid mode enabled. Click a point to place the box, then drag center, faces, corners, top, or rotate handle.');
    });
  }
  if (snapToGroundToggle) {
    snapToGroundToggle.addEventListener('change', () => {
      if (!state.selectionCuboid) return;
      updateCuboidGroundSnap();
      debounceCuboidStats();
      updateCuboidMeta();
      draw();
    });
  }

  clearPicksButton.addEventListener('click', () => {
    state.pickedPoints = [];
    state.selectionBounds = null;
    state.selectionCuboid = null;
    state.cuboidStats = null;
    state.selectionKind = 'polygon';
    state.selectionBox = null;
    updatePickUI();
    draw();
    setStatus('Selection cleared.');
  });

  analyzeButton.addEventListener('click', analyze);
  if (refreshFilesButton) refreshFilesButton.addEventListener('click', refreshFiles);
  loadPreviewButton.addEventListener('click', loadPreview);
  fileSelect.addEventListener('change', () => {
    state.currentFile = fileSelect.value;
    setStatus(`Selected dataset: ${state.currentFile}. Click Load Preview.`);
  });

  window.addEventListener('keydown', (event) => {
    if (event.key === '1') setInteractionMode('navigate');
    if (event.key === '2') setInteractionMode('point');
    if (event.key === '3') setInteractionMode('box');
    if (event.key === '4') setInteractionMode('cuboid');
    if (event.key === 'Delete') {
      state.selectionCuboid = null;
      state.cuboidStats = null;
      state.selectionKind = 'polygon';
      updatePickUI();
      draw();
    }
  });

  updatePickUI();
  resizeCanvas();
  if (state.currentFile) {
    fileSelect.value = state.currentFile;
    loadPreview();
  } else {
    setStatus('No .ply or .pcd files found in the workspace.', true);
  }
})();
