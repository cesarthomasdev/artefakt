/* =====================================================================
 * ASCII FLOW FIELD
 * Recriação do efeito de hero do artefakt.mov
 *
 * Como funciona (3 etapas):
 *   1. O texto é desenhado num canvas 2D invisível e cada pixel opaco
 *      vira uma partícula (nuvem de pontos com o formato da palavra).
 *   2. As partículas são simuladas na GPU (ping-pong de render targets):
 *      um flow field de simplex noise 4D as empurra, elas "morrem" e
 *      renascem na posição original, e o mouse as repele — é isso que
 *      faz os caracteres se dissiparem sob o cursor.
 *   3. O resultado é renderizado num buffer pequeno (30% da tela) e um
 *      shader de pós-processamento troca cada célula desse buffer por um
 *      caractere de um atlas, escolhido pela luminância da célula.
 *
 * Requer um elemento com id="ascii-hero" na página.
 * Texto: atributo data-text no elemento, ou CFG.text abaixo.
 *
 * Para usar um LOGO em vez de texto: dentro de sampleText(), troque
 * ctx.fillText(...) por ctx.drawImage(img, x, y, w, h) com um PNG/SVG
 * de silhueta branca sobre fundo transparente.
 * ===================================================================== */

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js';

/* ------------------------------------------------------------------ *
 *  CONFIG
 * ------------------------------------------------------------------ */
const CFG = {
  text:            'ARTEFAKT',
  font:            '900 {size}px "Arial Black", Impact, sans-serif',
  fontSizeVW:      13,          // tamanho da fonte em % da largura da tela
  sampleStep:      2,           // menor = mais partículas = mais denso
  intensity:       1.5,        // brilho de cada partícula (controla qual caractere aparece)
  asciiChars:      ' .:-=+*#%@4RT3F',
  columns:         180,         // nº de colunas de caracteres (90 no mobile)
  renderScale:     0.3,         // escala do buffer interno (0.9 no mobile)
  contrast:        1.09,
  brightness:      0.0,
  flowInfluence:   0.43,
  flowStrength:    1.09,
  flowFrequency:   0.53,
  lifeSpeed:       0.32,
  mouseRadius:     0.22,        // fração da altura da tela
  mouseStrength:   2.6,
  pointSize:       1.5
};

/* ------------------------------------------------------------------ *
 *  GLSL: simplex noise 4D (Ashima / Stefan Gustavson — MIT)
 * ------------------------------------------------------------------ */
const NOISE = /* glsl */`
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
float permute(float x){return floor(mod(((x*34.0)+1.0)*x, 289.0));}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
float taylorInvSqrt(float r){return 1.79284291400159 - 0.85373472095314 * r;}
vec4 grad4(float j, vec4 ip){
  const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
  vec4 p,s;
  p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
  p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
  s = vec4(lessThan(p, vec4(0.0)));
  p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;
  return p;
}
float simplexNoise4d(vec4 v){
  const vec2 C = vec2(0.138196601125010504, 0.309016994374947451);
  vec4 i  = floor(v + dot(v, C.yyyy));
  vec4 x0 = v -   i + dot(i, C.xxxx);
  vec4 i0;
  vec3 isX  = step(x0.yzw, x0.xxx);
  vec3 isYZ = step(x0.zww, x0.yyz);
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;
  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;
  vec4 i3 = clamp(i0, 0.0, 1.0);
  vec4 i2 = clamp(i0-1.0, 0.0, 1.0);
  vec4 i1 = clamp(i0-2.0, 0.0, 1.0);
  vec4 x1 = x0 - i1 + 1.0 * C.xxxx;
  vec4 x2 = x0 - i2 + 2.0 * C.xxxx;
  vec4 x3 = x0 - i3 + 3.0 * C.xxxx;
  vec4 x4 = x0 - 1.0 + 4.0 * C.xxxx;
  i = mod(i, 289.0);
  float j0 = permute(permute(permute(permute(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute(permute(permute(permute(
             i.w + vec4(i1.w, i2.w, i3.w, 1.0))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0));
  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0);
  vec4 p0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  p4 *= taylorInvSqrt(dot(p4,p4));
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)), 0.0);
  m0 = m0 * m0; m1 = m1 * m1;
  return 49.0 * (dot(m0*m0, vec3(dot(p0,x0), dot(p1,x1), dot(p2,x2)))
               + dot(m1*m1, vec2(dot(p3,x3), dot(p4,x4))));
}`;

/* ------------------------------------------------------------------ *
 *  BOOT
 * ------------------------------------------------------------------ */
const root = document.getElementById('ascii-hero');
if (root.dataset.text) CFG.text = root.dataset.text;

const isMobile = window.matchMedia('(max-width: 767px)').matches;
if (isMobile) { CFG.columns = 90; CFG.renderScale = 0.9; CFG.fontSizeVW = 20; }

let W = root.clientWidth, H = root.clientHeight;

const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(W, H);
renderer.setClearColor(0x000000, 0);
root.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-W/2, W/2, H/2, -H/2, -5000, 5000);

/* --- 1. amostra o texto num canvas 2D → nuvem de pontos ------------ */
function sampleText(text, width, height) {
  const c = document.createElement('canvas');
  c.width = width; c.height = height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const size = Math.round(width * CFG.fontSizeVW / 100);
  ctx.fillStyle = '#fff';
  ctx.font = CFG.font.replace('{size}', size);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, width/2, height/2);

  const data = ctx.getImageData(0, 0, width, height).data;
  const pts = [];
  const s = CFG.sampleStep;
  for (let y = 0; y < height; y += s) {
    for (let x = 0; x < width; x += s) {
      if (data[(y*width + x)*4 + 3] > 128) {
        pts.push(x - width/2, -(y - height/2), (Math.random()-0.5) * 30);
      }
    }
  }
  return pts;
}

const pts   = sampleText(CFG.text, W, H);
const count = pts.length / 3;
const fboSize = Math.ceil(Math.sqrt(count));

/* --- 2. textura-base com as posições de origem --------------------- */
const baseData = new Float32Array(fboSize * fboSize * 4);
for (let i = 0; i < fboSize * fboSize; i++) {
  if (i < count) {
    baseData[i*4+0] = pts[i*3+0];
    baseData[i*4+1] = pts[i*3+1];
    baseData[i*4+2] = pts[i*3+2];
    baseData[i*4+3] = Math.random();          // offset de "vida"
  } else {
    baseData[i*4+3] = -1.0;                   // slot vazio
  }
}
const baseTexture = new THREE.DataTexture(baseData, fboSize, fboSize, THREE.RGBAFormat, THREE.FloatType);
baseTexture.needsUpdate = true;

/* --- 3. ping-pong FBO: simulação das partículas -------------------- */
const rtOpts = {
  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  format: THREE.RGBAFormat, type: THREE.FloatType, depthBuffer: false
};
let rtA = new THREE.WebGLRenderTarget(fboSize, fboSize, rtOpts);
let rtB = new THREE.WebGLRenderTarget(fboSize, fboSize, rtOpts);

const simScene  = new THREE.Scene();
const simCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const simMaterial = new THREE.ShaderMaterial({
  uniforms: {
    uParticles:     { value: null },
    uBase:          { value: baseTexture },
    uTime:          { value: 0 },
    uDelta:         { value: 0 },
    uFlowInfluence: { value: CFG.flowInfluence },
    uFlowStrength:  { value: CFG.flowStrength },
    uFlowFrequency: { value: CFG.flowFrequency },
    uLifeSpeed:     { value: CFG.lifeSpeed },
    uMouse:         { value: new THREE.Vector3(0, 0, 0) },
    uMouseRadius:   { value: H * CFG.mouseRadius },
    uMouseStrength: { value: CFG.mouseStrength },
    uScale:         { value: 1 }
  },
  vertexShader: `
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
  `,
  fragmentShader: NOISE + `
    uniform sampler2D uParticles;
    uniform sampler2D uBase;
    uniform float uTime, uDelta, uLifeSpeed;
    uniform float uFlowInfluence, uFlowStrength, uFlowFrequency;
    uniform vec3  uMouse;
    uniform float uMouseRadius, uMouseStrength, uScale;
    varying vec2 vUv;

    void main(){
      vec4 particle = texture2D(uParticles, vUv);
      vec4 base     = texture2D(uBase, vUv);

      if (base.a < 0.0) { gl_FragColor = vec4(0.0, 0.0, 0.0, -1.0); return; }

      if (particle.a >= 1.0) {
        particle.a   = fract(particle.a);
        particle.xyz = base.xyz;
      } else {
        float freq = uFlowFrequency * 0.006 / uScale;

        // força do fluxo: varia por partícula
        float strength = simplexNoise4d(vec4(base.xyz * freq * 0.4, uTime + 1.0));
        float influence = (uFlowInfluence - 0.5) * (-2.0);
        strength = smoothstep(influence, 1.0, strength);

        vec3 flow = vec3(
          simplexNoise4d(vec4(particle.xyz * freq + 0.0, uTime * 0.25)),
          simplexNoise4d(vec4(particle.xyz * freq + 1.0, uTime * 0.25)),
          simplexNoise4d(vec4(particle.xyz * freq + 2.0, uTime * 0.25))
        );
        particle.xyz += normalize(flow) * uDelta * strength * uFlowStrength * 32.0 * uScale;

        // repulsão do mouse -> é isso que "dissipa" os caracteres
        vec2  d    = particle.xy - uMouse.xy;
        float dist = length(d);
        float f    = 1.0 - smoothstep(0.0, uMouseRadius, dist);
        particle.xy += normalize(d + 0.0001) * f * f * uMouseStrength * uMouseRadius * uDelta;

        particle.a += uDelta * uLifeSpeed;
      }
      gl_FragColor = particle;
    }
  `
});
simScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), simMaterial));

// estado inicial = posições-base
(function seed(){
  const init = new THREE.DataTexture(baseData.slice(), fboSize, fboSize, THREE.RGBAFormat, THREE.FloatType);
  init.needsUpdate = true;
  simMaterial.uniforms.uParticles.value = init;
  simMaterial.uniforms.uDelta.value = 0;
  renderer.setRenderTarget(rtA); renderer.render(simScene, simCamera);
  renderer.setRenderTarget(rtB); renderer.render(simScene, simCamera);
  renderer.setRenderTarget(null);
})();

/* --- 4. as partículas em si --------------------------------------- */
const geo = new THREE.BufferGeometry();
const refs  = new Float32Array(count * 2);
const seeds = new Float32Array(count);
for (let i = 0; i < count; i++) {
  refs[i*2+0] = (i % fboSize) / fboSize + 0.5 / fboSize;
  refs[i*2+1] = Math.floor(i / fboSize) / fboSize + 0.5 / fboSize;
  seeds[i] = 0.55 + Math.random() * 0.45;
}
geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
geo.setAttribute('aRef',  new THREE.BufferAttribute(refs, 2));
geo.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));

const particles = new THREE.Points(geo, new THREE.ShaderMaterial({
  uniforms: {
    uParticles: { value: null },
    uPointSize: { value: CFG.pointSize },
    uIntensity: { value: CFG.intensity },
    uTime:      { value: 0 },
    uScale:     { value: renderer.getPixelRatio() * CFG.renderScale }
  },
  vertexShader: NOISE + `
    uniform sampler2D uParticles;
    uniform float uPointSize, uScale, uIntensity, uTime;
    attribute vec2 aRef;
    attribute float aSeed;
    varying float vAlpha;
    void main(){
      vec4 p = texture2D(uParticles, aRef);
      if (p.a < 0.0) { gl_Position = vec4(2.0); vAlpha = 0.0; return; }
      float fadeIn  = smoothstep(0.0, 0.12, p.a);
      float fadeOut = 1.0 - smoothstep(0.85, 1.0, p.a);
      // campo de "luz" que desliza: é o que faz os caracteres mudarem sozinhos
      float shade = simplexNoise4d(vec4(p.xyz * 0.006, uTime * 0.12)) * 0.5 + 0.5;
      vAlpha = fadeIn * fadeOut * aSeed * (0.22 + shade * 1.05) * uIntensity;
      vec4 mv = modelViewMatrix * vec4(p.xyz, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = uPointSize * uScale;
    }
  `,
  fragmentShader: `
    varying float vAlpha;
    void main(){ gl_FragColor = vec4(vec3(1.0) * vAlpha, vAlpha); }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending
}));
particles.frustumCulled = false;
scene.add(particles);

/* --- 5. atlas de caracteres (canvas 2D -> textura) ----------------- */
function makeAsciiTexture(chars) {
  const c = document.createElement('canvas');
  c.width = 16 * chars.length; c.height = 16;
  const ctx = c.getContext('2d');
  ctx.fillStyle = 'black'; ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = 'white'; ctx.font = '16px monospace';
  ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
  chars.split('').forEach((ch, i) => ctx.fillText(ch, 16 * (i + 0.5), 8));
  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  return tex;
}

/* --- 6. passe de pós-processamento ASCII --------------------------- */
let bufW = Math.round(W * CFG.renderScale);
let bufH = Math.round(H * CFG.renderScale);
let sceneRT = new THREE.WebGLRenderTarget(bufW, bufH, {
  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter
});

const asciiMaterial = new THREE.ShaderMaterial({
  transparent: true,
  uniforms: {
    tDiffuse:         { value: sceneRT.texture },
    uResolution:      { value: new THREE.Vector2(bufW, bufH) },
    uAsciiTexture:    { value: makeAsciiTexture(CFG.asciiChars) },
    uCharCount:       { value: new THREE.Vector2(CFG.asciiChars.length, 1) },
    uAsciiPixelSize:  { value: bufW / CFG.columns },
    uAsciiBrightness: { value: CFG.brightness },
    uAsciiContrast:   { value: CFG.contrast },
    uAsciiMin:        { value: 0 },
    uAsciiMax:        { value: 1 }
  },
  vertexShader: `
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform vec2  uResolution;
    uniform float uAsciiPixelSize;
    uniform sampler2D uAsciiTexture;
    uniform vec2  uCharCount;
    uniform float uAsciiContrast, uAsciiBrightness, uAsciiMin, uAsciiMax;
    varying vec2 vUv;

    void main() {
      vec2 normalizedPixelSize = uAsciiPixelSize / uResolution;
      vec2 uvPixel = normalizedPixelSize * floor(vUv / normalizedPixelSize);
      vec4 texColor = texture2D(tDiffuse, uvPixel);

      float luma = dot(vec3(0.2126, 0.7152, 0.0722), texColor.rgb);

      luma = (luma - uAsciiMin) / (uAsciiMax - uAsciiMin);
      luma = clamp(luma, 0.0, 1.0);
      luma = luma + uAsciiBrightness;
      luma = (luma - 0.5) * uAsciiContrast + 0.5;
      luma = clamp(luma, 0.0, 1.0);

      vec2 cellUV = fract(vUv / normalizedPixelSize);

      float charIndex = clamp(
        floor(luma * (uCharCount.x - 1.0)),
        0.0,
        uCharCount.x - 1.0
      );

      vec2 asciiUV = vec2((charIndex + cellUV.x) / uCharCount.x, cellUV.y);
      float character = texture2D(uAsciiTexture, asciiUV).r;

      vec3  finalColor = character * vec3(1.0) * (luma + 0.9);
      float alpha = texColor.a * character;

      gl_FragColor = vec4(finalColor, alpha);
    }
  `
});
const asciiScene = new THREE.Scene();
asciiScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), asciiMaterial));

/* --- 7. mouse ------------------------------------------------------ */
const mouse = new THREE.Vector2(-9999, -9999);
const mouseLerp = new THREE.Vector2(-9999, -9999);
window.addEventListener('pointermove', (e) => {
  const r = root.getBoundingClientRect();
  mouse.set(e.clientX - r.left - W/2, -(e.clientY - r.top - H/2));
}, { passive: true });
root.addEventListener('pointerleave', () => mouse.set(-9999, -9999));

/* --- 8. loop ------------------------------------------------------- */
const clock = new THREE.Clock();
let prev = 0;
function tick() {
  const t  = clock.getElapsedTime();
  const dt = Math.min(t - prev, 1/30);
  prev = t;

  mouseLerp.lerp(mouse, 0.12);
  simMaterial.uniforms.uMouse.value.set(mouseLerp.x, mouseLerp.y, 0);
  simMaterial.uniforms.uTime.value  = t;
  simMaterial.uniforms.uDelta.value = dt;
  simMaterial.uniforms.uParticles.value = rtA.texture;

  renderer.setRenderTarget(rtB);
  renderer.render(simScene, simCamera);
  renderer.setRenderTarget(null);
  [rtA, rtB] = [rtB, rtA];

  particles.material.uniforms.uParticles.value = rtA.texture;
  particles.material.uniforms.uTime.value = t;

  renderer.setRenderTarget(sceneRT);
  renderer.clear();
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);

  renderer.render(asciiScene, simCamera);
  requestAnimationFrame(tick);
}
tick();

/* --- 9. resize ----------------------------------------------------- */
let rzT;
window.addEventListener('resize', () => {
  clearTimeout(rzT);
  rzT = setTimeout(() => {
    W = root.clientWidth; H = root.clientHeight;
    renderer.setSize(W, H);
    camera.left = -W/2; camera.right = W/2; camera.top = H/2; camera.bottom = -H/2;
    camera.updateProjectionMatrix();
    bufW = Math.round(W * CFG.renderScale);
    bufH = Math.round(H * CFG.renderScale);
    sceneRT.setSize(bufW, bufH);
    asciiMaterial.uniforms.uResolution.value.set(bufW, bufH);
    asciiMaterial.uniforms.uAsciiPixelSize.value = bufW / CFG.columns;
    simMaterial.uniforms.uMouseRadius.value = H * CFG.mouseRadius;
  }, 200);
});
