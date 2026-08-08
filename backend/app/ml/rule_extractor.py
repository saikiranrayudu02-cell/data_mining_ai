from typing import List, Any, Optional

class RuleExtractor:
    """
    Traverses custom ID3 and J48 Decision Tree structures recursively
    to extract human-readable IF-THEN rules in a structured multi-line format.
    """
    
    @staticmethod
    def extract_rules(clf: Any, feature_names: List[str], target_attribute: str = "class") -> List[str]:
        """
        Extract logical IF-THEN rule paths from a decision tree model.
        Format matches:
        IF Feature1=Value1
        AND Feature2=Value2
        THEN Target=Prediction
        """
        rules = []
        if not hasattr(clf, "root") or clf.root is None:
            return rules
            
        root = clf.root
        
        # Capitalize target and features for clean presentation (e.g. Play, class -> Play)
        target_display = target_attribute[0].upper() + target_attribute[1:] if len(target_attribute) > 0 else "Class"
        
        def traverse(node: Any, current_path: List[str]):
            if node.is_leaf:
                # Format prediction value (capitalize yes/no or classes)
                pred_val = str(node.prediction)
                pred_display = pred_val[0].upper() + pred_val[1:] if len(pred_val) > 0 else "Unknown"
                
                if current_path:
                    path_str = "\nAND ".join(current_path)
                    rules.append(f"IF {path_str}\nTHEN {target_display}={pred_display}")
                else:
                    rules.append(f"THEN {target_display}={pred_display}")
                return
                
            feat = node.feature
            feat_display = feat[0].upper() + feat[1:] if len(feat) > 0 else "Feature"
            
            # Check for continuous splits (J48)
            if getattr(node, "is_continuous", False):
                threshold = node.threshold
                
                # Left branch (<= threshold)
                if True in node.children:
                    traverse(node.children[True], current_path + [f"{feat_display}<={threshold:.2f}"])
                # Right branch (> threshold)
                if False in node.children:
                    traverse(node.children[False], current_path + [f"{feat_display}>{threshold:.2f}"])
            else:
                # Nominal splits
                for val, child in node.children.items():
                    val_str = str(val)
                    val_display = val_str[0].upper() + val_str[1:] if len(val_str) > 0 else "Value"
                    traverse(child, current_path + [f"{feat_display}={val_display}"])
                    
        traverse(root, [])
        return rules
