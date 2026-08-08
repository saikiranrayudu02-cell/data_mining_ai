from typing import Any, Dict, List, Optional, Tuple
from app.schemas.dataset import DecisionTreeStructure, DecisionTreeNode, DecisionTreeEdge
from graphviz import Source
from app.utils.logger import get_logger

logger = get_logger(__name__)

class TreeVisualizer:
    """
    Translates custom decision tree node objects (J48Node, ID3Node)
    into WEKA-style Graphviz DOT code and hierarchical React Flow canvas nodes.
    Supports WEKA oval split nodes, soft green leaf nodes, and leaf count annotations.
    """
    
    @staticmethod
    def compile_dot_to_png(dot_string: str, output_path: str):
        """
        Compile Graphviz DOT layout code to a static PNG image on disk.
        """
        try:
            logger.info(f"Rendering Graphviz DOT to PNG at {output_path}")
            src = Source(dot_string)
            src.render(outfile=output_path, format="png", cleanup=True)
        except FileNotFoundError as fnf:
            logger.warning(
                f"Failed to find graphviz dot command on system PATH: {str(fnf)}. "
                "Skipping tree PNG compilation."
            )
        except Exception as e:
            logger.error(f"Failed to compile decision tree DOT image: {str(e)}")

    @staticmethod
    def get_tree_depth(node: Any) -> int:
        if node is None or node.is_leaf:
            return 0
        if not node.children:
            return 0
        return 1 + max(TreeVisualizer.get_tree_depth(child) for child in node.children.values())

    @staticmethod
    def get_leaf_count(node: Any) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return sum(TreeVisualizer.get_leaf_count(child) for child in node.children.values())

    @staticmethod
    def get_split_count(node: Any) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + sum(TreeVisualizer.get_split_count(child) for child in node.children.values())

    @staticmethod
    def to_graphviz(clf: Any, feature_names: List[str]) -> str:
        """
        Export tree to Graphviz DOT string formatted with WEKA visual styling:
        - Split nodes: Ovals (`shape=ellipse`), light blue fill (`#e0f2fe`), dark border (`#0284c7`)
        - Leaf nodes: Rounded boxes (`shape=box, style="filled,rounded"`), light green fill (`#dcfce7`), dark border (`#16a34a`)
        - Leaf text: Predicted Class + WEKA instance count annotation e.g. `Iris-setosa (50.0)` or `yes (9.0/0.0)`
        """
        if not hasattr(clf, "root") or clf.root is None:
            return "digraph Tree { }"
            
        root = clf.root
        dot_lines = [
            "digraph Tree {",
            "  graph [splines=polyline, nodesep=0.5, ranksep=0.8];",
            "  node [fontname=\"Helvetica-Bold\", fontsize=11];",
            "  edge [fontname=\"Helvetica\", fontsize=10, color=\"#64748b\", fontcolor=\"#0f172a\"];"
        ]
        node_counter = 0
        
        def traverse(node: Any, parent_idx: Optional[int] = None, label: Optional[str] = None):
            nonlocal node_counter
            current_idx = node_counter
            node_counter += 1
            
            if node.is_leaf:
                total = getattr(node, "total_instances", 0.0)
                err = getattr(node, "error_instances", 0.0)
                err_str = f"/{err:g}" if err > 0 else ""
                count_str = f" ({total:g}{err_str})" if total > 0 else ""
                leaf_text = f"{node.prediction}{count_str}"
                
                dot_lines.append(
                    f'  {current_idx} [label="{leaf_text}", shape=box, style="filled,rounded", fillcolor="#dcfce7", color="#16a34a", fontcolor="#14532d"];'
                )
            else:
                dot_lines.append(
                    f'  {current_idx} [label="{node.feature}", shape=ellipse, style=filled, fillcolor="#e0f2fe", color="#0284c7", fontcolor="#0369a1"];'
                )
                
            if parent_idx is not None:
                edge_label_str = f' [label="{label}"]' if label else ""
                dot_lines.append(f'  {parent_idx} -> {current_idx}{edge_label_str};')
                
            if not node.is_leaf:
                if getattr(node, "is_continuous", False):
                    thresh_str = f"{node.threshold:.2f}".rstrip('0').rstrip('.')
                    if True in node.children:
                        traverse(node.children[True], current_idx, f"<= {thresh_str}")
                    if False in node.children:
                        traverse(node.children[False], current_idx, f"> {thresh_str}")
                else:
                    for val, child in node.children.items():
                        traverse(child, current_idx, f"= {val}")
                        
        traverse(root)
        dot_lines.append("}")
        return "\n".join(dot_lines)

    @staticmethod
    def to_react_flow(clf: Any, feature_names: List[str]) -> DecisionTreeStructure:
        """
        Traverse custom tree node structure and compute WEKA hierarchical
        subtree layout coordinates for rendering in React Flow.
        """
        nodes: List[DecisionTreeNode] = []
        edges: List[DecisionTreeEdge] = []
        
        if not hasattr(clf, "root") or clf.root is None:
            return DecisionTreeStructure(nodes=nodes, edges=edges)
            
        root = clf.root
        
        def _get_leaf_width(node: Any) -> int:
            if node.is_leaf or not getattr(node, "children", None):
                return 1
            return sum(_get_leaf_width(child) for child in node.children.values())
            
        total_leaf_width = _get_leaf_width(root)
        node_counter = 0
        
        UNIT_X = 220 # Horizontal unit spacing per leaf range
        UNIT_Y = 130 # Vertical level spacing
        
        def _layout(node: Any, depth: int, left_leaf_offset: int, parent_id: Optional[str] = None, edge_label: Optional[str] = None):
            nonlocal node_counter
            node_counter += 1
            current_id = f"node-{node_counter}"
            
            w = _get_leaf_width(node)
            center_leaf = left_leaf_offset + w / 2.0
            x = center_leaf * UNIT_X
            y = depth * UNIT_Y
            
            if node.is_leaf:
                node_type = "leaf"
                total = getattr(node, "total_instances", 0.0)
                err = getattr(node, "error_instances", 0.0)
                err_str = f"/{err:g}" if err > 0 else ""
                label = f"{node.prediction} ({total:g}{err_str})" if total > 0 else str(node.prediction)
            else:
                node_type = "split"
                label = str(node.feature)
                
            nodes.append(DecisionTreeNode(
                id=current_id,
                type=node_type,
                label=label,
                position={"x": int(x), "y": int(y)},
                data={
                    "type": node_type,
                    "label": label,
                    "feature": getattr(node, "feature", None),
                    "prediction": getattr(node, "prediction", None),
                    "total_instances": getattr(node, "total_instances", 0.0),
                    "error_instances": getattr(node, "error_instances", 0.0)
                }
            ))
            
            if parent_id is not None:
                edges.append(DecisionTreeEdge(
                    id=f"edge-{parent_id}-{current_id}",
                    source=parent_id,
                    target=current_id,
                    label=edge_label
                ))
                
            if not node.is_leaf:
                curr_offset = left_leaf_offset
                if getattr(node, "is_continuous", False):
                    thresh_str = f"{node.threshold:.2f}".rstrip('0').rstrip('.')
                    if True in node.children:
                        child_w = _get_leaf_width(node.children[True])
                        _layout(node.children[True], depth + 1, curr_offset, current_id, f"<= {thresh_str}")
                        curr_offset += child_w
                    if False in node.children:
                        _layout(node.children[False], depth + 1, curr_offset, current_id, f"> {thresh_str}")
                else:
                    for val, child in node.children.items():
                        child_w = _get_leaf_width(child)
                        _layout(child, depth + 1, curr_offset, current_id, f"= {val}")
                        curr_offset += child_w
                        
        _layout(root, 0, 0)
        return DecisionTreeStructure(nodes=nodes, edges=edges)
