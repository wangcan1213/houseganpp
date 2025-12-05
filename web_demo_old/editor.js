// -----------------------------
// 全局变量
// -----------------------------
let nodes = [];
let links = [];
let nextId = 0;

// 当前是否处于“新增节点模式”
let addMode = null;  // e.g. "bedroom"

// 鼠标悬停检测距离（多少像素算“拖到了另一个节点”）
const SNAP_DISTANCE = 25;


// -----------------------------
// 初始化 SVG
// -----------------------------
const graphDiv = document.getElementById("graph");
const width = graphDiv.clientWidth;
const height = graphDiv.clientHeight;

const svg = d3.select("#graph")
  .append("svg")
  .attr("width", "100%")
  .attr("height", "100%")
  .attr("viewBox", `0 0 ${width} ${height}`);


// -----------------------------
// 力导向图
// -----------------------------
const simulation = d3.forceSimulation()
  .force("charge", d3.forceManyBody().strength(-240))
  .force("center", d3.forceCenter(width/2, height/2))
  .force("link", d3.forceLink().id(d => d.id).distance(120));

let linkGroup = svg.append("g")
  .attr("stroke", "#aaa")
  .attr("stroke-width", 2);

let nodeGroup = svg.append("g");
let labelGroup = svg.append("g");

let linkElements = linkGroup.selectAll("line");
let nodeElements = nodeGroup.selectAll("circle");
let labelElements = labelGroup.selectAll("text");


// -----------------------------
// 1) palette 图例设置
// -----------------------------
const ROOM_TYPES = [
  {label:"living", color:"#e63946"},
  {label:"kitchen", color:"#bc6c25"},
  {label:"bedroom", color:"#f1c40f"},
  {label:"bathroom", color:"#95a5a6"},
  {label:"balcony", color:"#7ed6df"},
  {label:"corridor", color:"#27ae60"},
  {label:"closet", color:"#8e44ad"}
];

// 在页面下方 palette 画按钮
const palette = d3.select("#palette");
ROOM_TYPES.forEach(rt => {

  palette.append("button")
    .attr("class", "room-btn")
    .style("background", rt.color)
    .text(rt.label)
    .on("click", () => {
      addMode = rt.label;
      document.getElementById("status").textContent =
        `点击画布任意位置创建节点：${rt.label}`;
    });
});


// -----------------------------
// 2) 点击画布创建节点
// -----------------------------
svg.on("click", (event) => {
  if (!addMode) return;

  const pt = d3.pointer(event);
  const newNode = {id: nextId++, label: addMode, x: pt[0], y: pt[1]};
  nodes.push(newNode);

  addMode = null;
  document.getElementById("status").textContent =
    "拖动节点可调整位置；拖动节点到另一节点上可添加/删除连线";

  updateGraph();
});


// -----------------------------
// 3) 节点拖拽逻辑 + “拖到另一个节点”自动添加/删除连线
// -----------------------------
function dragStarted(event, d) {
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(event, d) {
  d.fx = event.x;
  d.fy = event.y;
}

function dragEnded(event, d) {
  if (!event.active) simulation.alphaTarget(0);

  // 检查是否拖到了另一个节点上
  const target = findNearbyNode(d);

  if (target && target.id !== d.id) {
    toggleEdge(d.id, target.id);
  }

  d.fx = null;
  d.fy = null;
}

function findNearbyNode(sourceNode) {
  for (const n of nodes) {
    if (n.id === sourceNode.id) continue;
    const dx = n.x - sourceNode.x;
    const dy = n.y - sourceNode.y;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if (dist < SNAP_DISTANCE) return n;
  }
  return null;
}

function toggleEdge(id1, id2) {
  const idx = links.findIndex(
    e => (e.source.id ?? e.source) === id1 &&
         (e.target.id ?? e.target) === id2
  );

  // 如果已存在 → 删除
  if (idx >= 0) {
    links.splice(idx, 1);
  } else {
    // 添加连线
    links.push({source:id1, target:id2});
  }

  updateGraph();
}


// -----------------------------
// 4) 更新图形
// -----------------------------
function updateGraph() {

  linkElements = linkElements.data(links, d => d.source + "-" + d.target);
  linkElements.exit().remove();
  linkElements = linkElements.enter()
    .append("line")
    .merge(linkElements);

  nodeElements = nodeElements.data(nodes, d => d.id);
  nodeElements.exit().remove();
  nodeElements = nodeElements.enter()
    .append("circle")
    .attr("r", 16)
    .attr("fill", d => {
      let rt = ROOM_TYPES.find(r => r.label === d.label);
      return rt ? rt.color : "#ccc";
    })
    .attr("stroke", "#333")
    .attr("stroke-width", 2)
    .call(d3.drag()
      .on("start", dragStarted)
      .on("drag", dragged)
      .on("end", dragEnded)
    )
    .merge(nodeElements);

  labelElements = labelElements.data(nodes, d => d.id);
  labelElements.exit().remove();
  labelElements = labelElements.enter()
    .append("text")
    .attr("font-size", 11)
    .attr("dy", 4)
    .attr("text-anchor","middle")
    .text(d => d.label)
    .merge(labelElements);

  simulation.nodes(nodes).on("tick", ticked);
  simulation.force("link").links(links);

  simulation.alpha(0.7).restart();
}

function ticked() {
  linkElements
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);

  nodeElements
    .attr("cx", d => d.x)
    .attr("cy", d => d.y);

  labelElements
    .attr("x", d => d.x)
    .attr("y", d => d.y + 26);
}


// 初始化 living 节点
nodes.push({id: nextId++, label:"living", x: width/2, y: height/2});
updateGraph();
