import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

let WS_URL;

if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
  WS_URL = "ws://localhost:8000/ws";
} 
else {
  WS_URL = "wss://p01--regibert--ybpp4jlb5k9w.code.run/ws";
}

const MAX_LIVE_POSTS = 500;

let scene, camera, renderer, controls, composer;
let pointsSoutenuMesh = null;
let pointsCourantMesh = null;
let pointsFamilierMesh = null;
let boundingGroup = null; 
let raycaster, mouse;
let ws;

let livePostsData = [];
let globalStaticPoints = [];

const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const card = document.getElementById('card');
const postIdEl = document.getElementById('post-id');
const postTextEl = document.getElementById('post-text');
const barSoutenu = document.getElementById('bar-soutenu');
const barCourant = document.getElementById('bar-courant');
const barFamilier = document.getElementById('bar-familier');
const valSoutenu = document.getElementById('val-soutenu');
const valCourant = document.getElementById('val-courant');
const valFamilier = document.getElementById('val-familier');

const coordsHUD = document.createElement('div');
coordsHUD.id = 'coords-hud';
coordsHUD.style.position = 'absolute';
coordsHUD.style.bottom = '20px';
coordsHUD.style.right = '20px';
coordsHUD.style.color = '#475569';
coordsHUD.style.fontFamily = 'monospace';
coordsHUD.style.fontSize = '13px';
coordsHUD.style.pointerEvents = 'none';
coordsHUD.style.opacity = '0.7';
coordsHUD.style.textAlign = 'right';
document.body.appendChild(coordsHUD);

const style = document.createElement('style');
style.innerHTML = `
  .cyber-scroll::-webkit-scrollbar { width: 6px; }
  .cyber-scroll::-webkit-scrollbar-track { background: rgba(2, 4, 8, 0.4); }
  .cyber-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
  .cyber-scroll::-webkit-scrollbar-thumb:hover { background: #0ea5e9; }
`;
document.head.appendChild(style);

const projectInfoHUD = document.createElement('div');
projectInfoHUD.id = 'project-info';
projectInfoHUD.style.color = '#cbd5e1'; 
projectInfoHUD.style.fontFamily = 'monospace';
projectInfoHUD.style.pointerEvents = 'auto';
projectInfoHUD.style.background = 'rgba(2, 4, 8, 0.75)';
projectInfoHUD.style.padding = '18px';
projectInfoHUD.style.border = '1px solid #334155';
projectInfoHUD.style.borderRadius = '8px';
projectInfoHUD.style.backdropFilter = 'blur(6px)';
projectInfoHUD.style.width = '380px';
projectInfoHUD.style.boxShadow = '0 4px 30px rgba(0, 0, 0, 0.5)';
projectInfoHUD.style.position = 'absolute';
projectInfoHUD.style.bottom = '20px';
projectInfoHUD.style.left = '20px';
projectInfoHUD.style.zIndex = '50';

projectInfoHUD.innerHTML = `
  <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 12px;">
    <div>
      <div style="font-size: 19px; font-weight: bold; color: #0ea5e9; letter-spacing: 1px; margin-bottom: 2px;">
                <span class="material-icons" style="font-size: 22px;">info_outline</span>
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
      <a href="https://github.com/aogunleye" target="_blank" title="Voir le code source sur GitHub">
        <img src="github.png" alt="GitHub" style="width: 28px; height: 28px; opacity: 0.7; transition: 0.3s;" 
             onmouseover="this.style.opacity='1'; this.style.filter='drop-shadow(0 0 5px #0ea5e9)';" 
             onmouseout="this.style.opacity='0.7'; this.style.filter='none';">
      </a>
      <button id="info-close" class="panel-close" aria-label="Fermer" style="background:none; border:none; color:white; font-size:20px; cursor:pointer;">×</button>
    </div>
  </div>

  <div style="margin-bottom: 12px; padding: 6px 10px; background: rgba(14, 165, 233, 0.1); border: 1px dashed rgba(14, 165, 233, 0.4); border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
    <span style="font-size: 11px; font-weight: bold; color: #cbd5e1; letter-spacing: 0.5px;">FLUX FIREHOSE</span>
    <span style="font-size: 14px; font-weight: bold; color: #0ea5e9;"><span id="live-count">0</span> <span style="font-size: 11px; color: #64748b;">/ ${MAX_LIVE_POSTS}</span></span>
  </div>

  <div class="cyber-scroll" style="max-height: 40vh; overflow-y: auto; padding-right: 8px; font-size: 12px; line-height: 1.5; text-align: justify;">
    <p style="margin-top: 0;">
      Cette interface cartographie en temps réel le registre de langue des derniers posts francophones issues de l'<a href="https://docs.bsky.app/docs/advanced-guides/firehose" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: bold;">API Bluesky</a>.
    </p>
    
    <p>
      Au lieu de classer les textes dans des cases rigides, le modèle de langue neuronal <a href="https://huggingface.co/almanach/camembert-base" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: bold;">CamemBERT</a> a été entraîné sur le corpus <a href="http://tremolo.irisa.fr/fr/tremolo-tweets-corpus/" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: bold;">TREMoLo-Tweets</a> pour prédire une <i>distribution de probabilités</i>. 
      Les vecteurs à 768 dimensions sont ensuite projetés topologiquement en 3D via <b>UMAP</b>. J'ai nommé ce modèle <b>RegiBERT</b>.
    </p>
    <p>Pour plus d'informations sur ma démarche, les méthodes, les outils, les données, le modèle etc., consultez le <a href="https://github.com/aogunleye/registre-langue" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: bold;">dépôt GitHub</a>.</p>

    <div style="margin-top: 18px; margin-bottom: 8px; font-weight: bold; color: #0ea5e9; font-size: 12px; border-bottom: 1px dashed #334155; padding-bottom: 4px;">
      DESCRIPTION VISUELLE
    </div>
    
    <ul style="margin: 0; padding-left: 16px; color: #cbd5e1; font-size: 11.5px; line-height: 1.6;">
      <li style="margin-bottom: 6px;"><b>Le nuage de fond</b> représente les <span id="static-count" style="color: #0ea5e9; font-weight: bold;">...</span> tweets du jeu d'entraînement. Il est possible de filtrer l'affichage des nuages par couleur avec les boutons en haut.</li>
      <li style="margin-bottom: 6px;"><b>Les sphères néons</b> sont les posts du flux Bluesky en direct. Le double-clic redirige vers le post d'origine. Ces points sont les prédictions du modèle. Au bout de 500 posts, quand une nouvelle sphère apparaît, la plus ancienne disparaît.</li>
      <li style="margin-bottom: 6px;"><b>Les couleurs</b> sont déterminées par un mix des probabilités d'appartenance à chaque catégorie : <br><span style="color:#ef4444; font-weight:bold;">rouge = soutenu</span>, <span style="color:#3b82f6; font-weight:bold;">bleu = courant</span>, <span style="color:#10b981; font-weight:bold;">vert = familier</span>. Par exemple un post 20% familier, 40% courant et 40% soutenu aura une couleur plutôt mauve.</li>
      <li><b>Les coordonnées</b> délimitent le plus petit cube capable d'enfermer tous les points du jeu d'entraînement.</li>
    </ul>
  </div>
`;

projectInfoHUD.addEventListener('wheel', (e) => e.stopPropagation());

const filtersHUD = document.createElement('div');
filtersHUD.id = 'filters-hud';
filtersHUD.style.position = 'absolute';
filtersHUD.style.top = '20px';
filtersHUD.style.left = '50%';
filtersHUD.style.transform = 'translateX(-50%)';
filtersHUD.style.display = 'flex';
filtersHUD.style.gap = '15px';
filtersHUD.style.zIndex = '100';
filtersHUD.style.pointerEvents = 'auto';

function createFilterButton(label, colorHex, key) {
  const btn = document.createElement('button');
  btn.innerText = label;
  btn.style.background = 'rgba(2, 4, 8, 0.8)';
  btn.style.color = colorHex;
  btn.style.border = `1px solid ${colorHex}`;
  btn.style.padding = '6px 16px';
  btn.style.borderRadius = '4px';
  btn.style.cursor = 'pointer';
  btn.style.fontFamily = 'monospace';
  btn.style.fontWeight = 'bold';
  btn.style.transition = '0.3s';
  btn.style.backdropFilter = 'blur(4px)';
  
  let active = true;
  btn.onclick = () => {
    active = !active;
    btn.style.opacity = active ? '1' : '0.4';
    btn.style.background = active ? 'rgba(2, 4, 8, 0.8)' : 'transparent';
    if (key === 'soutenu' && pointsSoutenuMesh) pointsSoutenuMesh.visible = active;
    if (key === 'courant' && pointsCourantMesh) pointsCourantMesh.visible = active;
    if (key === 'familier' && pointsFamilierMesh) pointsFamilierMesh.visible = active;
  };
  return btn;
}

filtersHUD.appendChild(createFilterButton('SOUTENU', '#dc2626', 'soutenu'));
filtersHUD.appendChild(createFilterButton('COURANT', '#2563eb', 'courant'));
filtersHUD.appendChild(createFilterButton('FAMILIER', '#10b981', 'familier'));

document.body.appendChild(filtersHUD);
document.body.appendChild(projectInfoHUD);

function initScene() {
  const container = document.getElementById('canvas-container');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x020408);
  scene.fog = new THREE.FogExp2(0x020408, 0.012);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(25, 20, 35);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.useLegacyLights = false; 
  container.appendChild(renderer.domElement);

  const renderScene = new RenderPass(scene, camera);
  const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.2, 0.5, 0.5);
  
  composer = new EffectComposer(renderer);
  composer.addPass(renderScene);
  composer.addPass(bloomPass);

  const ambientLight = new THREE.AmbientLight(0x0f172a, 1.5);
  scene.add(ambientLight);
  const neonRed = new THREE.PointLight(0xdc2626, 200, 100); neonRed.position.set(10, 0, 0); scene.add(neonRed);
  const neonBlue = new THREE.PointLight(0x2563eb, 200, 100); neonBlue.position.set(-10, 0, 10); scene.add(neonBlue);
  const neonGreen = new THREE.PointLight(0x10b981, 200, 100); neonGreen.position.set(0, 0, -10); scene.add(neonGreen);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.rotateSpeed = 0.8;

  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.5;
  mouse = new THREE.Vector2(-100, -100);

  window.addEventListener('resize', onWindowResize);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('dblclick', onDoubleClick);

  animate();
}

function createCoordSprite(text, x, y, z) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = '22px monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillStyle = '#475569';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.7, depthTest: false });
  const sprite = new THREE.Sprite(spriteMat);
  sprite.position.set(x, y, z);
  sprite.scale.set(6, 1.5, 1);
  return sprite;
}

function createSubCloud(arr) {
  const positions = new Float32Array(arr.length * 3);
  const colors = new Float32Array(arr.length * 3);
  for(let i=0; i<arr.length; i++) {
    positions[i*3] = arr[i][0]; positions[i*3+1] = arr[i][1]; positions[i*3+2] = arr[i][2];
    if (arr[i].length >= 6) {
      colors[i*3] = arr[i][3]; colors[i*3+1] = arr[i][4]; colors[i*3+2] = arr[i][5];
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.PointsMaterial({ size: 0.10, vertexColors: true, transparent: true, opacity: 0.15, sizeAttenuation: true });
  return new THREE.Points(geo, mat);
}

function createBackgroundCloud(pointsData) {
    const countSpan = document.getElementById('static-count');
  if (countSpan) {
    countSpan.textContent = new Intl.NumberFormat('fr-FR').format(pointsData.length);
  }
  if (pointsSoutenuMesh) scene.remove(pointsSoutenuMesh, pointsCourantMesh, pointsFamilierMesh);
  if (boundingGroup) scene.remove(boundingGroup);

  globalStaticPoints = pointsData;
  let arrS = [], arrC = [], arrF = [];
  let allPositions = [];

  for (let i = 0; i < pointsData.length; i++) {
    const p = pointsData[i];
    allPositions.push(p[0], p[1], p[2]);
    let r = p[3] || 0.2, g = p[4] || 0.25, b = p[5] || 0.3;
    
    if (r > g && r > b) arrS.push(p);
    else if (b > r && b > g) arrC.push(p);
    else arrF.push(p);
  }

  pointsSoutenuMesh = createSubCloud(arrS);
  pointsCourantMesh = createSubCloud(arrC);
  pointsFamilierMesh = createSubCloud(arrF);
  
  scene.add(pointsSoutenuMesh);
  scene.add(pointsCourantMesh);
  scene.add(pointsFamilierMesh);

  const dummyGeo = new THREE.BufferGeometry();
  dummyGeo.setAttribute('position', new THREE.Float32BufferAttribute(allPositions, 3));
  dummyGeo.computeBoundingBox();
  const bbox = dummyGeo.boundingBox;
  const dummyMesh = new THREE.Points(dummyGeo);
  
  boundingGroup = new THREE.Group();
  const boxHelper = new THREE.BoxHelper(dummyMesh, 0x334155);
  boxHelper.material.transparent = true; boxHelper.material.opacity = 0.25; 
  boundingGroup.add(boxHelper);

  const sizeX = bbox.max.x - bbox.min.x;
  const sizeZ = bbox.max.z - bbox.min.z;
  const gridHelper = new THREE.GridHelper(Math.max(sizeX, sizeZ), 20, 0x334155, 0x334155);
  gridHelper.position.y = bbox.min.y;
  gridHelper.position.x = (bbox.max.x + bbox.min.x) / 2;
  gridHelper.position.z = (bbox.max.z + bbox.min.z) / 2;
  gridHelper.material.transparent = true; gridHelper.material.opacity = 0.15;
  boundingGroup.add(gridHelper);

  const corners = [
    [bbox.min.x, bbox.min.y, bbox.min.z], [bbox.max.x, bbox.min.y, bbox.min.z],
    [bbox.min.x, bbox.max.y, bbox.min.z], [bbox.max.x, bbox.max.y, bbox.min.z],
    [bbox.min.x, bbox.min.y, bbox.max.z], [bbox.max.x, bbox.min.y, bbox.max.z],
    [bbox.min.x, bbox.max.y, bbox.max.z], [bbox.max.x, bbox.max.y, bbox.max.z]
  ];
  corners.forEach(c => {
    const text = `[${c[0].toFixed(0)}, ${c[1].toFixed(0)}, ${c[2].toFixed(0)}]`;
    boundingGroup.add(createCoordSprite(text, c[0] * 1.05, c[1] * 1.05, c[2] * 1.05));
  });
  scene.add(boundingGroup);
  dummyGeo.dispose();
}

function addLivePostMarker(postData) {
  if (livePostsData.length >= MAX_LIVE_POSTS) {
    const oldestMarker = livePostsData.shift();
    scene.remove(oldestMarker);
    oldestMarker.geometry.dispose();
    oldestMarker.material.dispose();
  }

  const [x, y, z] = postData.coords;
  const [r, g, b] = postData.rgb;
  const color = new THREE.Color(`rgb(${r}, ${g}, ${b})`);

  const geometry = new THREE.SphereGeometry(0.175, 32, 32); 
  const material = new THREE.MeshStandardMaterial({ 
    color: color,
    roughness: 0.2,
    metalness: 0.8,
    emissive: color,
    emissiveIntensity: 1.2
  });

  const marker = new THREE.Mesh(geometry, material);
  marker.position.set(x, y, z);
  marker.userData = postData;

  scene.add(marker);
  livePostsData.push(marker);

  const ringGeo = new THREE.RingGeometry(0.2, 0.3, 32);
  const ringMat = new THREE.MeshBasicMaterial({
    color: color, side: THREE.DoubleSide, transparent: true, opacity: 1
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.position.set(x, y, z);
  ring.lookAt(camera.position);
  scene.add(ring);

  let scale = 1;
  let opacity = 1;
  const expandRing = setInterval(() => {
    scale += 0.15; opacity -= 0.05;
    ring.scale.set(scale, scale, scale);
    ringMat.opacity = opacity;
    if (opacity <= 0) {
      clearInterval(expandRing);
      scene.remove(ring); ringGeo.dispose(); ringMat.dispose();
    }
  }, 30);

    const liveCountSpan = document.getElementById('live-count');
    if (liveCountSpan) {
        liveCountSpan.textContent = livePostsData.length;
    }
    
    updateHUDCard(postData);
}

function connectWebSocket() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => { statusBadge.className = 'badge connected'; statusText.textContent = 'En direct'; };
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'init_background') {
      createBackgroundCloud(data.points);
    } else if (data.type === 'new_post') {
      addLivePostMarker(data);
    }
  };
  
  ws.onclose = () => { statusBadge.className = 'badge disconnected'; statusText.textContent = 'Déconnecté'; setTimeout(connectWebSocket, 3000); };
  ws.onerror = (err) => { ws.close(); };
}

function updateHUDCard(data) {
  if (!data) return;
  
  if (window.innerWidth > 800) {
    card.classList.remove('hidden');
  }

  let dateText = "";
  if (data.created_at) {
    const d = new Date(data.created_at);
    dateText = d.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' }) + 
               ' à ' + 
               d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
  }

  const shortAuthor = data.author ? data.author.replace('did:plc:', '').substring(0, 8) + '...' : 'anonyme';

  const postIdEl = document.getElementById('post-id');
  const postTypeEl = document.getElementById('post-type');
  
  if (postIdEl) postIdEl.textContent = `@${shortAuthor}`;
  if (postTypeEl) postTypeEl.textContent = dateText;
  
  postTextEl.textContent = `"${data.text}"`;

  const pSoutenu = Math.round(data.probs.soutenu * 100);
  const pCourant = Math.round(data.probs.courant * 100);
  const pFamilier = Math.round(data.probs.familier * 100);

  barSoutenu.style.width = `${pSoutenu}%`; 
  barCourant.style.width = `${pCourant}%`; 
  barFamilier.style.width = `${pFamilier}%`;
  
  valSoutenu.textContent = `${pSoutenu}%`; 
  valCourant.textContent = `${pCourant}%`; 
  valFamilier.textContent = `${pFamilier}%`;
}

function onPointerMove(event) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
}

function checkIntersections() {
  if (livePostsData.length === 0) return;
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(livePostsData);
  if (intersects.length > 0) {
    document.body.style.cursor = 'pointer';
    updateHUDCard(intersects[0].object.userData);
    if (window.innerWidth <= 800) {
       card.classList.remove('hidden');
       card.classList.add('mobile-open');
    }
  } else {
    document.body.style.cursor = 'default';
  }
}

function onDoubleClick(event) {
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(livePostsData);
  if (intersects.length > 0) {
    const data = intersects[0].object.userData;
    
    if (data.url) {
      window.open(data.url, '_blank');
    } else {
      const query = encodeURIComponent(`"${data.text}"`);
      window.open(`https://bsky.app/search?q=${query}`, '_blank');
    }
  }
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  checkIntersections();
  
  if (coordsHUD) {
    coordsHUD.innerHTML = `COORD [CAM]<br>X: ${camera.position.x.toFixed(2)}<br>Y: ${camera.position.y.toFixed(2)}<br>Z: ${camera.position.z.toFixed(2)}`;
  }

  composer.render();
}

initScene();
connectWebSocket();

const waitingMessage = document.getElementById('waiting-message');
const countdownSpan = document.getElementById('countdown');
let countdownValue = 5;

const startTimer = setInterval(() => {
  countdownValue--;
  if (countdownValue > 0) {
    if (countdownSpan) countdownSpan.textContent = countdownValue;
  } else {
    clearInterval(startTimer);
    if (waitingMessage) {
      waitingMessage.innerHTML = "Les nouveaux posts s'affichent toutes les 5 secondes.";
      waitingMessage.style.background = "rgba(2, 4, 8, 0.7)";
      waitingMessage.style.border = "1px solid #334155";
    }
  }
}, 1000);

const infoTab = document.getElementById('info-tab');
const cardTab = document.getElementById('card-tab');
const infoClose = document.getElementById('info-close');
const projectInfoElement = document.getElementById('project-info');

if (infoTab) {
  infoTab.addEventListener('click', () => {
    projectInfoElement.classList.toggle('mobile-open');
    card.classList.remove('mobile-open'); 
  });
}

if (infoClose) {
  infoClose.addEventListener('click', () => {
    projectInfoElement.classList.remove('mobile-open');
  });
}

if (cardTab) {
  cardTab.addEventListener('click', () => {
    card.classList.toggle('mobile-open');
    if (projectInfoElement) projectInfoElement.classList.remove('mobile-open'); 
  });
}