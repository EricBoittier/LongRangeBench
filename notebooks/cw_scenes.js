// Three.js scenes for the ChargeWorld notebook (chargeworld.py), ported from the "Charge Learning Under Scrutiny" scenes.
// Visual grammar: colour = charge sign (blue negative, red positive, near-white zero); shape = role (sphere = atom,
// octahedron = region charge, arrow = force or dipole, thin rod = coupling); line thickness = coupling strength;
// translucent = not fixed by the data. Labels are HTML overlays.
// THREE is loaded by the Python wrapper (cw_widgets.py) from the vendored, pinned r170 build: no network at run time.
let THREE = null;

function col(hex) { return new THREE.Color(hex); }
function mix(a, b, t) { return a.clone().lerp(b, Math.max(0, Math.min(1, t))); }
function chargeColour(G, q, qmax) {
  // intensity grows with sqrt(|q| / qmax), so ±0.5 e stays visible next to ±3 e
  const t = Math.sign(q) * Math.sqrt(Math.min(1, Math.abs(q) / (qmax || 1)));
  return t >= 0 ? mix(col(G.zero), col(G.pos), t) : mix(col(G.zero), col(G.neg), -t);
}
function mat(colour, opacity = 1) {
  return new THREE.MeshStandardMaterial({ color: colour, roughness: 0.55, metalness: 0.0, transparent: opacity < 1,
    opacity, depthWrite: opacity >= 1 });
}
let SPHERE = null;
function ball(pos, r, colour, opacity = 1) {
  const m = new THREE.Mesh(SPHERE, mat(colour, opacity)); m.position.set(...pos); m.scale.setScalar(r); return m;
}
function rod(a, b, radius, colour, opacity = 1) {
  const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b), d = B.clone().sub(A), L = d.length();
  const m = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, L, 10, 1), mat(colour, opacity));
  m.position.copy(A).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  return m;
}
function arrow(from, vec, colour, shaft = 0.05, opacity = 1) {
  const g = new THREE.Group(), V = new THREE.Vector3(...vec), L = V.length();
  if (L < 1e-6) return g;
  const head = Math.min(0.45, 0.4 * L), A = new THREE.Vector3(...from), u = V.clone().normalize();
  const tip = A.clone().addScaledVector(u, L - head);
  if (L - head > 1e-3) g.add(rod(A.toArray(), tip.toArray(), shaft, colour, opacity));
  const cone = new THREE.Mesh(new THREE.ConeGeometry(shaft * 2.6, head, 14), mat(colour, opacity));
  cone.position.copy(tip).addScaledVector(u, head / 2);
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), u);
  g.add(cone); return g;
}
function clear(group) {
  while (group.children.length) {
    const c = group.children.pop();
    c.traverse((o) => { if (o.geometry && o.geometry !== SPHERE) o.geometry.dispose(); if (o.material) o.material.dispose(); });
  }
}

// ------------------------------------------------------------------------------------------- stage
function makeStage(el, G, opts) {
  const W = opts.width || 760, H = opts.height || Math.round(W * 0.6);
  const wrap = document.createElement("div");
  wrap.style.cssText = `position:relative;width:100%;max-width:${W}px;font-family:${G.font};color:${G.ink};`;
  const box = document.createElement("div");
  box.style.cssText = `position:relative;width:100%;aspect-ratio:${W}/${H};background:${G.card};border-radius:6px;overflow:hidden;touch-action:none;cursor:grab;`;
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(col(G.card)); renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.style.cssText = "width:100%;height:100%;display:block;";
  const overlay = document.createElement("div");
  overlay.style.cssText = "position:absolute;inset:0;pointer-events:none;font-size:12px;";
  box.append(renderer.domElement, overlay);
  const note = document.createElement("div");
  note.style.cssText = "font-size:12.5px;color:#4d4d4d;padding:5px 2px 0;min-height:1.2em;line-height:1.4;";
  wrap.append(box, note); el.appendChild(wrap);
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d8d0, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.55); key.position.set(6, 10, 8); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.45); fill.position.set(-8, 4, -6); scene.add(fill);
  const camera = new THREE.PerspectiveCamera(30, W / H, 0.1, 2000);
  const orbit = { az: THREE.MathUtils.degToRad(-35), el: THREE.MathUtils.degToRad(22), dist: opts.distance || 30,
    target: new THREE.Vector3(0, 0, 0) };
  const root = new THREE.Group(); scene.add(root);
  const labels = [];
  function place() {
    const t = orbit.target, c = Math.cos(orbit.el);
    camera.position.set(t.x + orbit.dist * c * Math.sin(orbit.az), t.y + orbit.dist * Math.sin(orbit.el), t.z + orbit.dist * c * Math.cos(orbit.az));
    camera.lookAt(t);
  }
  function draw() {
    const w = box.clientWidth || W, h = box.clientHeight || H;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    renderer.render(scene, camera);
    for (const L of labels) {
      const p = L.pos.clone().project(camera);
      L.el.style.left = ((p.x + 1) / 2 * 100) + "%"; L.el.style.top = ((1 - p.y) / 2 * 100) + "%";
      L.el.style.display = p.z < 1 ? "" : "none";
    }
  }
  let drag = null;
  box.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; box.setPointerCapture(e.pointerId); });
  box.addEventListener("pointerup", () => { drag = null; });
  box.addEventListener("pointermove", (e) => {
    if (!drag) return;
    orbit.az -= (e.clientX - drag[0]) * 0.008; orbit.el = Math.max(-1.3, Math.min(1.3, orbit.el + (e.clientY - drag[1]) * 0.006));
    drag = [e.clientX, e.clientY]; place(); draw();
  });
  box.addEventListener("wheel", (e) => { e.preventDefault(); orbit.dist *= Math.exp(e.deltaY * 0.001); place(); draw(); }, { passive: false });
  function label(pos, html, size = 12, colour = null) {
    const d = document.createElement("div"); d.innerHTML = html;
    d.style.cssText = `position:absolute;white-space:nowrap;transform:translate(-50%,-50%);color:${colour || G.ink};font-size:${size}px;`;
    overlay.appendChild(d); labels.push({ pos: new THREE.Vector3(...pos), el: d });
  }
  function clearLabels() { labels.splice(0).forEach((L) => L.el.remove()); }
  window.addEventListener("resize", draw);
  return { scene, root, camera, renderer, orbit, place, draw, label, clearLabels, note, box };
}

// ------------------------------------------------------------------------------------------- 1. structure explorer
// data: {R:[[x,y,z]], q:[...], alpha:[...]?, halo:[bool]?, cell:[9]?, arrows:[[from,vec]]?, arrow_colour?, frags:[{c,Q}]?,
//        qmax, note, distance?, labels:[[pos, html]]?}
function explorerScene(S, D, G) {
  const n = D.R.length, qmax = D.qmax || 3;
  const ctr = D.centre || [0, 0, 0];
  const P = D.R.map((x) => [x[0] - ctr[0], x[1] - ctr[1], x[2] - ctr[2]]);
  const rscale = D.atom_scale || 1;
  P.forEach((x, i) => {
    const q = D.q[i], a = D.alpha ? D.alpha[i] : (Math.abs(q) < 0.05 ? 0.35 : 1);
    S.root.add(ball(x, rscale * (0.28 + 0.2 * Math.sqrt(Math.abs(q))), chargeColour(G, q, qmax), a));
    if (D.halo && D.halo[i]) S.root.add(ball(x, rscale * (0.62 + 0.2 * Math.sqrt(Math.abs(q))), col(G.accent), 0.28));
  });
  if (D.cell) {
    const a = D.cell.slice(0, 3), b = D.cell.slice(3, 6), c = D.cell.slice(6, 9), o = [-ctr[0], -ctr[1], -ctr[2]];
    const pt = (i, j, k) => [0, 1, 2].map((d) => o[d] + i * a[d] + j * b[d] + k * c[d]);
    const E = [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 1, 0]], [[1, 0, 0], [1, 0, 1]],
      [[0, 1, 0], [1, 1, 0]], [[0, 1, 0], [0, 1, 1]], [[0, 0, 1], [1, 0, 1]], [[0, 0, 1], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]],
      [[1, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 1]]];
    for (const [u, v] of E) S.root.add(rod(pt(...u), pt(...v), 0.05, col(G.greys[1]), 0.6));
  }
  if (D.arrows) for (const [i, v] of D.arrows) S.root.add(arrow(P[i], v, col(D.arrow_colour || G.ink), 0.06));
  if (D.frags) for (const f of D.frags) {
    const p = [f.c[0] - ctr[0], f.c[1] - ctr[1] + (D.frag_lift || 0), f.c[2] - ctr[2]];
    const o = new THREE.Mesh(new THREE.OctahedronGeometry(0.6 + 0.5 * Math.sqrt(Math.abs(f.Q))), mat(chargeColour(G, f.Q, qmax), 0.9));
    o.position.set(...p); S.root.add(o);
    S.label([p[0], p[1] + 1.6, p[2]], `region ${f.Q >= 0 ? "+" : "−"}${Math.abs(f.Q).toFixed(2)} e`, 12);
  }
  if (D.labels) for (const [p, h] of D.labels) S.label(p, h, 12, G.greys[0]);
}

// ------------------------------------------------------------------------------------------- 2. tier ladder
// data: {tiers:[{tier,label,X,q}]}
function ladderScene(S, D, G) {
  const T = D.tiers, k0 = (T.length - 1) / 2;
  T.forEach((t, k) => {
    const x = (k - k0) * 4.0, y = (k - k0) * 0.55;
    const plinth = new THREE.Mesh(new THREE.BoxGeometry(3.3, 0.22, 3.3), mat(col(G.greys[3])));
    plinth.position.set(x, y - 0.15, 0); S.root.add(plinth);
    const ext = Math.max(...t.X.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1]), Math.abs(p[2])))) || 1, sc = 1.45 / ext;
    t.X.forEach((p, i) => {
      const q = t.q[i];
      S.root.add(ball([x + p[0] * sc, y + 1.55 + p[1] * sc, p[2] * sc], 0.07 + 0.075 * Math.min(1.7, Math.sqrt(Math.abs(q))),
        chargeColour(G, q, 1.5), Math.abs(q) < 0.05 ? 0.35 : 1));
    });
    S.label([x, y - 0.8, 1.9], `<b>${t.tier}</b>`, 12);
  });
}

// ------------------------------------------------------------------------------------------- 3. species graph
// data: {q0:[12], names:[12], colour_q:[12], edges:[[i,j,w,kind]], qmax}; kind 0 = shared structures, 1 = added (combo)
function graphScene(S, D, G) {
  const n = D.q0.length;
  const P = D.q0.map((q, i) => {   // ring ordered by reference charge; height = reference charge
    const f = 2 * Math.PI * i / n; return [6.0 * Math.cos(f), 0.9 * q, 6.0 * Math.sin(f)];
  });
  for (const [i, j, w, kind] of D.edges) {
    if (w <= 0) continue;
    S.root.add(rod(P[i], P[j], 0.012 + 0.11 * Math.sqrt(w), col(kind === 1 ? G.accent : G.greys[1]), kind === 1 ? 0.95 : 0.55));
  }
  D.q0.forEach((q, i) => {
    const cq = D.colour_q ? D.colour_q[i] : q, fixed = !D.translucent || !D.translucent[i];
    S.root.add(ball(P[i], 0.3 + 0.28 * Math.sqrt(Math.abs(q)), chargeColour(G, cq, D.qmax || 3), fixed ? 1 : 0.3));
    S.label([P[i][0] * 1.17, P[i][1] + 0.85, P[i][2] * 1.17], D.names[i], 12);
  });
}

// ------------------------------------------------------------------------------------------- 4. loss landscape
// data: {a:[n], L:[[n x n]], runs:[{traj:[[a,b]], trapped}], marks:[[a,b,html]]}
function landscapeScene(S, D, G) {
  const a = D.a, L = D.L, n = a.length;
  const lmax = D.lcap || 40;
  const sx = (v) => v * 3.0, sz = (v) => -v * 3.0, ly = (v) => 3.2 * Math.log10(1 + Math.min(v, lmax)) - 1.5;
  const geo = new THREE.BufferGeometry(), pos = [], cols = [], idx = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    pos.push(sx(a[i]), ly(L[i][j]), sz(a[j]));
    const t = Math.min(1, Math.log10(1 + Math.min(L[i][j], lmax)) / Math.log10(1 + lmax));
    const c = mix(col(G.greys[3]), col(G.greys[0]), t); cols.push(c.r, c.g, c.b);
  }
  for (let i = 0; i < n - 1; i++) for (let j = 0; j < n - 1; j++) { const k = i * n + j; idx.push(k, k + n, k + 1, k + 1, k + n, k + n + 1); }
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
  geo.setIndex(idx); geo.computeVertexNormals();
  S.root.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, side: THREE.DoubleSide, transparent: true, opacity: 0.9 })));
  S.root.add(new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: col(G.greys[2]), transparent: true, opacity: 0.12 })));
  const Lat = (u, v) => {
    const fi = (u - a[0]) / (a[1] - a[0]), fj = (v - a[0]) / (a[1] - a[0]);
    const i = Math.max(0, Math.min(n - 2, Math.floor(fi))), j = Math.max(0, Math.min(n - 2, Math.floor(fj)));
    const s = Math.min(1, Math.max(0, fi - i)), t = Math.min(1, Math.max(0, fj - j));
    return (1 - s) * (1 - t) * L[i][j] + s * (1 - t) * L[i + 1][j] + (1 - s) * t * L[i][j + 1] + s * t * L[i + 1][j + 1];
  };
  const prog = D.progress ?? 1;
  for (const r of D.runs) {
    const T = r.traj, upto = Math.max(2, Math.round(prog * T.length));
    const pts = T.slice(0, upto).map(([u, v]) => new THREE.Vector3(sx(u), ly(Lat(u, v)) + 0.08, sz(v)));
    // path colour: purple = ends trapped (sign-mixed), green = ends at the exact charges (not a charge colour)
    const c = r.trapped ? col(G.trap) : col(G.ok);
    if (pts.length > 2) S.root.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), Math.min(300, pts.length * 3), 0.035, 6, false), mat(c, 0.85)));
    const e = T[upto - 1]; S.root.add(ball([sx(e[0]), ly(Lat(e[0], e[1])) + 0.12, sz(e[1])], 0.12, c));
  }
  for (const [u, v, h] of (D.marks || [])) S.label([sx(u), ly(Lat(u, v)) + 1.1, sz(v)], h, 12);
}

const SCENES = { explorer: explorerScene, ladder: ladderScene, graph: graphScene, landscape: landscapeScene };

function build(S, model) {
  const D = model.get("data") || {}, G = model.get("grammar"), p = model.get("params") || {};
  clear(S.root); S.clearLabels();
  SCENES[model.get("scene")](S, D, G);
  if (D.distance && !S._userZoom) { S.orbit.dist = D.distance; }
  if (p.azimuth !== undefined && !S._moved) S.orbit.az = THREE.MathUtils.degToRad(p.azimuth);
  if (p.elevation !== undefined && !S._moved) S.orbit.el = THREE.MathUtils.degToRad(p.elevation);
  S.note.innerHTML = D.note || "";
  S.place(); S.draw();
}

export default {
  async render({ model, el }) {
    if (!THREE) THREE = await loadThree();
    if (!SPHERE) SPHERE = new THREE.SphereGeometry(1, 24, 16);
    const p = model.get("params") || {};
    const S = makeStage(el, model.get("grammar"), { width: p.width, height: p.height, distance: (model.get("data") || {}).distance });
    S.box.addEventListener("pointerdown", () => { S._moved = true; });
    S.box.addEventListener("wheel", () => { S._userZoom = true; });
    build(S, model);
    model.on("change:data", () => build(S, model));
    model.on("change:params", () => build(S, model));
  },
};
