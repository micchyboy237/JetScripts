// Build element tree
function buildTree(element, depth = 0) {
  if (depth > 5) return null; // Limit depth

  const node = {
    tag: element.tagName.toLowerCase(),
    id: element.id || null,
    classes: Array.from(element.classList),
    role: element.getAttribute("role") || null,
    ariaLabel: element.getAttribute("aria-label") || null,
    text: element.innerText?.trim().substring(0, 100) || null,
    children: [],
    attributes: {},
    isVisible: element.offsetParent !== null,
    rect: {
      x: element.getBoundingClientRect().left,
      y: element.getBoundingClientRect().top,
      width: element.getBoundingClientRect().width,
      height: element.getBoundingClientRect().height,
    },
  };

  // Get important attributes
  const importantAttrs = [
    "href",
    "src",
    "alt",
    "title",
    "name",
    "type",
    "value",
  ];
  for (const attr of importantAttrs) {
    if (element.hasAttribute(attr)) {
      node.attributes[attr] = element.getAttribute(attr);
    }
  }

  // Recursively process children
  for (const child of element.children) {
    const childNode = buildTree(child, depth + 1);
    if (childNode) {
      node.children.push(childNode);
    }
  }

  return node;
}

// Focus on specific area if requested
let rootElement = document.body;
const areaSelectors = {
  header: 'header, [role="banner"], .header, #header',
  main: 'main, [role="main"], .main, #main, .content',
  footer: 'footer, [role="contentinfo"], .footer, #footer',
  sidebar: 'aside, [role="complementary"], .sidebar, #sidebar',
  navigation: 'nav, [role="navigation"], .nav, #nav',
};

if ("{focus_area}" in areaSelectors) {
  const focused = document.querySelector(areaSelectors["{focus_area}"]);
  if (focused) rootElement = focused;
}

buildTree(rootElement);
