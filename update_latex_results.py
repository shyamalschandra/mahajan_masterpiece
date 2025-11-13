"""
Helper script to update LaTeX documents with actual benchmark results.
This script reads benchmark_results.json and updates the LaTeX files.
"""

import json
import re
import sys
from pathlib import Path


def load_results(json_path: str = 'benchmark_results.json') -> dict:
    """Load benchmark results from JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found. Please run benchmark.py first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_path}")
        sys.exit(1)


def format_number(value: float, decimals: int = 4) -> str:
    """Format number for LaTeX."""
    return f"{value:.{decimals}f}"


def format_time(value: float, decimals: int = 2) -> str:
    """Format time value for LaTeX."""
    return f"{value:.{decimals}f}"


def update_presentation(results: dict):
    """Update presentation.tex with results."""
    ff_results = results['feedforward_nn']
    trans_results = results['transformer']
    
    # Read presentation file
    with open('presentation.tex', 'r') as f:
        content = f.read()
    
    # Update performance metrics table
    content = re.sub(
        r'Accuracy & \\textbf\{0\.XXXX\} & \\textbf\{0\.XXXX\}',
        f'Accuracy & \\textbf{{{format_number(ff_results["accuracy"])}}} & \\textbf{{{format_number(trans_results["accuracy"])}}}',
        content
    )
    content = re.sub(
        r'Precision & 0\.XXXX & 0\.XXXX',
        f'Precision & {format_number(ff_results["precision"])} & {format_number(trans_results["precision"])}',
        content
    )
    content = re.sub(
        r'Recall & 0\.XXXX & 0\.XXXX',
        f'Recall & {format_number(ff_results["recall"])} & {format_number(trans_results["recall"])}',
        content
    )
    content = re.sub(
        r'F1 Score & 0\.XXXX & 0\.XXXX',
        f'F1 Score & {format_number(ff_results["f1_score"])} & {format_number(trans_results["f1_score"])}',
        content
    )
    
    # Update efficiency table
    content = re.sub(
        r'Training Time \(s\) & \\textbf\{XX\.XX\} & \\textbf\{XX\.XX\}',
        f'Training Time (s) & \\textbf{{{format_time(ff_results["train_time"])}}} & \\textbf{{{format_time(trans_results["train_time"])}}}',
        content
    )
    content = re.sub(
        r'Inference Time \(ms\) & X\.XXXX & X\.XXXX',
        f'Inference Time (ms) & {format_number(ff_results["inference_time"] * 1000, 2)} & {format_number(trans_results["inference_time"] * 1000, 2)}',
        content
    )
    content = re.sub(
        r'Parameters & X,XXX & XXX,XXX',
        f'Parameters & {ff_results["num_parameters"]:,} & {trans_results["num_parameters"]:,}',
        content
    )
    
    # Write updated content
    with open('presentation.tex', 'w') as f:
        f.write(content)
    
    print("Updated presentation.tex with benchmark results")


def update_paper(results: dict):
    """Update paper.tex with results."""
    ff_results = results['feedforward_nn']
    trans_results = results['transformer']
    
    # Read paper file
    with open('paper.tex', 'r') as f:
        content = f.read()
    
    # Update performance metrics table
    content = re.sub(
        r'Accuracy & \\textbf\{0\.XXXX\} & \\textbf\{0\.XXXX\}',
        f'Accuracy & \\textbf{{{format_number(ff_results["accuracy"])}}} & \\textbf{{{format_number(trans_results["accuracy"])}}}',
        content
    )
    content = re.sub(
        r'Precision & 0\.XXXX & 0\.XXXX',
        f'Precision & {format_number(ff_results["precision"])} & {format_number(trans_results["precision"])}',
        content
    )
    content = re.sub(
        r'Recall & 0\.XXXX & 0\.XXXX',
        f'Recall & {format_number(ff_results["recall"])} & {format_number(trans_results["recall"])}',
        content
    )
    content = re.sub(
        r'F1 Score & 0\.XXXX & 0\.XXXX',
        f'F1 Score & {format_number(ff_results["f1_score"])} & {format_number(trans_results["f1_score"])}',
        content
    )
    
    # Update efficiency table
    content = re.sub(
        r'Training Time \(seconds\) & \\textbf\{XX\.XX\} & \\textbf\{XX\.XX\}',
        f'Training Time (seconds) & \\textbf{{{format_time(ff_results["train_time"])}}} & \\textbf{{{format_time(trans_results["train_time"])}}}',
        content
    )
    content = re.sub(
        r'Inference Time \(milliseconds\) & X\.XXXX & X\.XXXX',
        f'Inference Time (milliseconds) & {format_number(ff_results["inference_time"] * 1000, 2)} & {format_number(trans_results["inference_time"] * 1000, 2)}',
        content
    )
    content = re.sub(
        r'Number of Parameters & X,XXX & XXX,XXX',
        f'Number of Parameters & {ff_results["num_parameters"]:,} & {trans_results["num_parameters"]:,}',
        content
    )
    
    # Write updated content
    with open('paper.tex', 'w') as f:
        f.write(content)
    
    print("Updated paper.tex with benchmark results")


def main():
    """Main function."""
    print("Loading benchmark results...")
    results = load_results()
    
    print("\nUpdating LaTeX documents...")
    update_presentation(results)
    update_paper(results)
    
    print("\nDone! You can now compile the LaTeX documents:")
    print("  pdflatex presentation.tex")
    print("  pdflatex paper.tex")


if __name__ == "__main__":
    main()

