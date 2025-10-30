import re
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Node:
    condition: Optional[str]  # None for root
    marginal: float
    entropy: float
    children: List['Node']
    depth: int
    
    def __str__(self, prefix=""):
        if self.condition is None:
            s = "[ROOT]\n"
        else:
            s = f"[{self.condition}] marginal={self.marginal:.4f} entropy={self.entropy:.4f}\n"
        for i, child in enumerate(self.children):
            is_last = i == len(self.children) - 1
            connector = "├── " if not is_last else "└── "
            child_prefix = prefix + ("│   " if not is_last else "    ")
            s += prefix + connector + child.__str__(child_prefix)
        return s

def parse_logs(log_text):
    lines = log_text.strip().split('\n')
    pattern_full = r'### branching on (.+?); marginal=([\d.]+); entropy=([\d.]+)'
    pattern_partial = r'### branching on (.+?)$'
    
    decisions = []
    for line in lines:
        match = re.match(pattern_full, line)
        if match:
            condition = match.group(1)
            marginal = float(match.group(2))
            entropy = float(match.group(3))
        else:
            match = re.match(pattern_partial, line)
            if match:
                condition = match.group(1)
                marginal = 0.0
                entropy = 0.0
            else:
                continue
        decisions.append((condition, marginal, entropy))
    
    if not decisions:
        return None
    
    root = Node(None, 0.0, 0.0, [], 0)
    current = root
    nodes_by_condition = {}  # Map positive conditions to their nodes
    
    for condition, marginal, entropy in decisions:
        if '!=' in condition:
            # Negation: backtrack to parent of matching positive
            var_name = extract_variable(condition)
            positive_cond = condition.replace('!=', '=')
            
            # Find the positive node
            if positive_cond in nodes_by_condition:
                positive_node = nodes_by_condition[positive_cond]
                # Backtrack to its parent
                current = find_parent(root, positive_node)
        
        # Create new child node
        child = Node(condition, marginal, entropy, [], current.depth + 1)
        current.children.append(child)
        current = child
        
        # Store positive conditions
        if '!=' not in condition:
            nodes_by_condition[condition] = child
    
    return root

def extract_variable(condition):
    match = re.match(r'(x\[\d+,\d+\])', condition)
    return match.group(1) if match else None

def find_parent(root, target):
    """Find parent of target node"""
    if target in root.children:
        return root
    for child in root.children:
        result = find_parent(child, target)
        if result:
            return result
    return None

# Example usage
log_text = """### branching on x[4,5]=8
### branching on x[3,3]=18
### branching on x[3,3]!=18
### branching on x[3,3]=15
### branching on x[3,3]!=15
### branching on x[4,5]!=8
### branching on x[4,5]=3
### branching on x[4,5]!=3
### branching on x[4,5]=7
### branching on x[4,5]!=7
### branching on x[4,5]=6
### branching on x[4,5]!=6
### branching on x[3,4]=19"""

tree = parse_logs(log_text)
print(tree)
