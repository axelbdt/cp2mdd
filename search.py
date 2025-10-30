"""Parse CSP solver search traces into tree structure for BDD conversion."""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any


@dataclass
class Node:
    """Node in search tree representing a decision or root."""
    
    decision: Optional[Tuple[str, str, Any]]  # (variable, operator, value) or None for root
    label: Optional[str] = None  # 'SAT' | 'UNSAT' | None
    children: List['Node'] = field(default_factory=list)
    parent: Optional['Node'] = None
    depth: int = 0
    
    def is_leaf(self) -> bool:
        """Check if node is a labeled leaf."""
        return self.label is not None
    
    def path_from_root(self) -> List[Tuple[str, str, Any]]:
        """Collect all decisions from root to this node."""
        path = []
        node = self
        while node.parent is not None:
            if node.decision is not None:
                path.append(node.decision)
            node = node.parent
        return list(reversed(path))
    
    def __str__(self, prefix: str = "") -> str:
        """Pretty print tree structure."""
        if self.decision is None:
            s = "[ROOT]\n"
        else:
            var, op, val = self.decision
            label_str = f" [{self.label}]" if self.label else ""
            s = f"{var}{op}{val}{label_str}\n"
        
        for i, child in enumerate(self.children):
            is_last = i == len(self.children) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            s += prefix + connector + child.__str__(child_prefix)
        
        return s


@dataclass
class SearchTree:
    """Search tree built from solver trace."""
    
    root: Node
    
    @classmethod
    def from_log(cls, log_text: str) -> 'SearchTree':
        """Parse log text into search tree.
        
        Args:
            log_text: Solver trace with decision lines and leaf labels
            
        Returns:
            SearchTree with all paths from trace
            
        Raises:
            ValueError: If trace is malformed
        """
        decisions = parse_log_to_decisions(log_text)
        root = build_tree_from_decisions(decisions)
        return cls(root)
    
    def extract_all_paths(self) -> List[Tuple[List[Tuple[str, str, Any]], str]]:
        """Extract all paths from root to leaves with labels.
        
        Returns:
            List of (decision_sequence, label) pairs
        """
        return extract_paths(self.root, positive_only=False)
    
    def extract_sat_paths(self) -> List[List[Tuple[str, str, Any]]]:
        """Extract only paths leading to SAT leaves, positive decisions only.
        
        Returns:
            List of decision sequences (variable, operator, value)
        """
        all_paths = extract_paths(self.root, positive_only=True)
        return [path for path, label in all_paths if label == 'SAT']
    
    def extract_positive_only_paths(self) -> List[Tuple[List[Tuple[str, str, Any]], str]]:
        """Extract all paths with only positive decisions.
        
        Returns:
            List of (decision_sequence, label) pairs
        """
        return extract_paths(self.root, positive_only=True)
    
    def validate(self) -> None:
        """Validate all paths satisfy one-hot constraint.
        
        Raises:
            ValueError: If any path violates one-hot constraint
        """
        for path, label in self.extract_positive_only_paths():
            validate_path(path)
    
    def __str__(self) -> str:
        """Pretty print entire tree."""
        return str(self.root)


def parse_log_to_decisions(log_text: str) -> List[Tuple[Tuple[str, str, Any], Optional[str]]]:
    """Parse log text into flat list of decisions with optional labels.
    
    Args:
        log_text: Raw solver trace
        
    Returns:
        List of ((variable, operator, value), label) tuples
        where label is 'SAT', 'UNSAT', or None
    """
    pattern = r'### branching on (.+?)(=|!=)(.+)$'
    
    decisions = []
    lines = log_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        # Check for leaf label
        if line in ['SAT', 'UNSAT']:
            if not decisions:
                raise ValueError(f"Leaf label '{line}' without preceding decision")
            # Attach label to previous decision
            prev_decision, prev_label = decisions[-1]
            if prev_label is not None:
                raise ValueError(f"Decision already has label '{prev_label}', cannot add '{line}'")
            decisions[-1] = (prev_decision, line)
            continue
        
        # Parse decision
        match = re.match(pattern, line)
        if not match:
            raise ValueError(f"Could not parse line: {line}")
        
        var = match.group(1).strip()
        op = match.group(2)
        val_str = match.group(3).strip()
        
        # Try to parse value as int, fall back to string
        try:
            val = int(val_str)
        except ValueError:
            val = val_str
        
        decisions.append(((var, op, val), None))
    
    return decisions


def find_positive_ancestor(stack: List[Node], var: str, val: Any) -> Optional[int]:
    """Find index of most recent ancestor with decision (var, '=', val).
    
    Args:
        stack: Current path from root to current node
        var: Variable name to search for
        val: Value to match
        
    Returns:
        Index in stack, or None if not found
    """
    for i in reversed(range(len(stack))):
        node = stack[i]
        if node.decision is None:
            continue
        node_var, node_op, node_val = node.decision
        if node_var == var and node_op == '=' and node_val == val:
            return i
    return None


def build_tree_from_decisions(
    decisions: List[Tuple[Tuple[str, str, Any], Optional[str]]]
) -> Node:
    """Build search tree from flat decision list.
    
    Args:
        decisions: List of ((var, op, val), label) from parse_log_to_decisions
        
    Returns:
        Root node of constructed tree
        
    Raises:
        ValueError: If trace is malformed (e.g., negation without positive ancestor)
    """
    root = Node(decision=None, label=None, depth=0)
    current = root
    stack = [root]  # Path from root to current node
    
    for (var, op, val), label in decisions:
        
        if op == '=':
            # Positive decision: create child and descend
            child = Node(
                decision=(var, op, val),
                label=None,
                parent=current,
                depth=current.depth + 1
            )
            current.children.append(child)
            stack.append(child)
            current = child
            
            # Apply label if present
            if label:
                current.label = label
                # Backtrack after leaf
                stack.pop()
                current = stack[-1]
        
        elif op == '!=':
            # Negative decision: mark ancestor UNSAT, create sibling
            
            # Find positive ancestor
            ancestor_idx = find_positive_ancestor(stack, var, val)
            
            if ancestor_idx is None:
                raise ValueError(
                    f"Negation {var}!={val} without matching positive ancestor {var}={val}"
                )
            
            target_node = stack[ancestor_idx]
            
            # Mark as UNSAT if unlabeled
            if target_node.label is None:
                target_node.label = 'UNSAT'
            
            # Backtrack to target's parent
            stack = stack[:ancestor_idx]
            current = stack[-1]
            
            # Create sibling with negation
            child = Node(
                decision=(var, op, val),
                label=None,
                parent=current,
                depth=current.depth + 1
            )
            current.children.append(child)
            stack.append(child)
            current = child
            
            # Apply label if present
            if label:
                current.label = label
                stack.pop()
                current = stack[-1]
        
        else:
            raise ValueError(f"Unknown operator: {op}")
    
    return root


def extract_paths(
    node: Node,
    path_so_far: List[Tuple[str, str, Any]] = None,
    positive_only: bool = False
) -> List[Tuple[List[Tuple[str, str, Any]], str]]:
    """Recursively extract all paths from root to labeled leaves.
    
    Args:
        node: Current node in traversal
        path_so_far: Accumulated path from root
        positive_only: If True, only include positive (=) decisions in paths
        
    Returns:
        List of (path, label) tuples where path is list of decisions
    """
    if path_so_far is None:
        path_so_far = []
    
    # Add current decision to path
    if node.decision is not None:
        var, op, val = node.decision
        if positive_only:
            # Only include positive decisions
            if op == '=':
                path = path_so_far + [(var, op, val)]
            else:
                path = path_so_far
        else:
            # Include all decisions
            path = path_so_far + [(var, op, val)]
    else:
        path = path_so_far
    
    # Leaf node: return path with label
    if node.label is not None:
        return [(path, node.label)]
    
    # Internal node without children: assume UNSAT
    if not node.children:
        return [(path, 'UNSAT')]
    
    # Internal node: recurse on children
    paths = []
    for child in node.children:
        paths.extend(extract_paths(child, path, positive_only))
    
    return paths


def validate_path(path: List[Tuple[str, str, Any]]) -> None:
    """Validate that path satisfies one-hot constraint.
    
    Args:
        path: List of (variable, operator, value) decisions
        
    Raises:
        ValueError: If path violates one-hot (multiple positive assignments to same variable)
    """
    positive_assignments = {}
    
    for var, op, val in path:
        if op == '=':
            if var in positive_assignments:
                existing = positive_assignments[var]
                if existing != val:
                    raise ValueError(
                        f"Path violates one-hot constraint: "
                        f"{var}={existing} and {var}={val} both present"
                    )
            positive_assignments[var] = val


def main():
    """Test search tree parser on example trace."""
    
    log_text = """
### branching on x[4,5]=8
### branching on x[3,3]=14
### branching on x[3,3]!=14
### branching on x[3,3]=22
### branching on x[3,3]!=22
### branching on x[3,3]=15
### branching on x[3,3]!=15
### branching on x[4,5]!=8
### branching on x[4,5]=3
### branching on x[4,5]!=3
### branching on x[4,5]=7
### branching on x[4,5]!=7
### branching on x[4,5]=11
### branching on x[3,4]=19
SAT
"""
    
    print("Parsing search trace...")
    print("=" * 60)
    
    try:
        tree = SearchTree.from_log(log_text)
        
        print("\nTree structure:")
        print(tree)
        
        print("\nAll paths (with negations):")
        print("-" * 60)
        for i, (path, label) in enumerate(tree.extract_all_paths(), 1):
            path_str = " → ".join(f"{v}{op}{val}" for v, op, val in path)
            print(f"Path {i}: {path_str} [{label}]")
        
        print("\nPositive-only paths:")
        print("-" * 60)
        for i, (path, label) in enumerate(tree.extract_positive_only_paths(), 1):
            path_str = " → ".join(f"{v}={val}" for v, op, val in path)
            print(f"Path {i}: {path_str} [{label}]")
        
        print("\nSAT paths (for BDD encoding):")
        print("-" * 60)
        sat_paths = tree.extract_sat_paths()
        if sat_paths:
            for i, path in enumerate(sat_paths, 1):
                path_str = " ∧ ".join(f"{v}={val}" for v, op, val in path)
                print(f"Solution {i}: {path_str}")
        else:
            print("No SAT paths found")
        
        print("\nValidating paths...")
        tree.validate()
        print("✓ All paths satisfy one-hot constraint")
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
