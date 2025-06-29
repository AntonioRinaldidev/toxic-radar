# AI/gui/toxic_radar_gui.py
"""
ToxicRadar GUI - Native Python interface that calls microservices directly
instead of making HTTP API calls. Much faster and more efficient.
"""

from AI.core.system_detector import get_system_config
from AI.reasoning.reasoning import apply_reasoning
from AI.paraphraser.voting import custom_utility_score
from AI.paraphraser.scorer import score_toxicity, taunt_equivalence_score, score_fluency
from AI.paraphraser.generator import generate_paraphrases
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Direct imports from microservices


@dataclass
class ParaphraseResult:
    """Result structure matching the API response"""
    original: str
    candidates: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    fallback_used: bool = False
    fallback_message: str = ""


class ToxicRadarGUI:
    """Main GUI application for ToxicRadar"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ToxicRadar - Desktop Interface")
        self.root.geometry("900x700")

        # Get system configuration
        self.system_config = get_system_config()

        # Initialize variables
        self.is_processing = False
        self.current_result = None

        self.setup_ui()
        self.setup_styles()

    def setup_styles(self):
        """Configure custom styles for the interface"""
        style = ttk.Style()

        # Configure styles for different toxicity levels
        style.configure("Low.TLabel", foreground="green")
        style.configure("Medium.TLabel", foreground="orange")
        style.configure("High.TLabel", foreground="red")
        style.configure("Processing.TLabel", foreground="blue")

    def setup_ui(self):
        """Setup the main user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="🎯 ToxicRadar Desktop",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # System info
        system_info = f"System: {self.system_config.device.upper()} | " \
            f"OS: {self.system_config.os_type.title()}"
        ttk.Label(main_frame, text=system_info, font=("Arial", 9)).grid(
            row=1, column=0, columnspan=2, pady=(0, 10))

        # Input section
        input_frame = ttk.LabelFrame(
            main_frame, text="Input Text", padding="10")
        input_frame.grid(row=2, column=0, columnspan=2,
                         sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)

        # Text input area
        self.text_input = scrolledtext.ScrolledText(
            input_frame, height=6, wrap=tk.WORD)
        self.text_input.grid(
            row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.text_input.insert(
            tk.END, "Enter text here to analyze toxicity or generate paraphrases...")

        # Input controls
        controls_frame = ttk.Frame(input_frame)
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Number of candidates
        ttk.Label(controls_frame, text="Candidates:").grid(
            row=0, column=0, padx=(0, 5))
        self.num_candidates = tk.StringVar(value="3")
        candidates_spinbox = ttk.Spinbox(controls_frame, from_=1, to=10, width=5,
                                         textvariable=self.num_candidates)
        candidates_spinbox.grid(row=0, column=1, padx=(0, 20))

        # Generation mode
        ttk.Label(controls_frame, text="Mode:").grid(
            row=0, column=2, padx=(0, 5))
        self.mode = tk.StringVar(value="auto")
        mode_combo = ttk.Combobox(controls_frame, textvariable=self.mode,
                                  values=["auto", "efficient",
                                          "quality", "universal"],
                                  state="readonly", width=10)
        mode_combo.grid(row=0, column=3, padx=(0, 20))

        # Action buttons
        self.analyze_btn = ttk.Button(controls_frame, text="🔍 Analyze Only",
                                      command=self.analyze_toxicity)
        self.analyze_btn.grid(row=0, column=4, padx=(0, 10))

        self.paraphrase_btn = ttk.Button(controls_frame, text="✏️ Paraphrase",
                                         command=self.paraphrase_text)
        self.paraphrase_btn.grid(row=0, column=5)

        # Progress bar
        self.progress = ttk.Progressbar(input_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

        # Results section
        results_frame = ttk.LabelFrame(
            main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2,
                           sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Results notebook (tabs)
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.rowconfigure(0, weight=1)

        # Toxicity Analysis tab
        self.toxicity_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(
            self.toxicity_frame, text="📊 Toxicity Analysis")
        self.setup_toxicity_tab()

        # Paraphrases tab
        self.paraphrases_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(
            self.paraphrases_frame, text="✏️ Paraphrases")
        self.setup_paraphrases_tab()

    def setup_toxicity_tab(self):
        """Setup the toxicity analysis tab"""
        self.toxicity_frame.columnconfigure(0, weight=1)

        # Overall status
        self.status_label = ttk.Label(self.toxicity_frame, text="No analysis yet",
                                      font=("Arial", 12, "bold"))
        self.status_label.grid(row=0, column=0, pady=(0, 10))

        # Toxicity scores frame
        scores_frame = ttk.LabelFrame(
            self.toxicity_frame, text="Toxicity Scores", padding="10")
        scores_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        scores_frame.columnconfigure(1, weight=1)

        # Score labels (will be populated dynamically)
        self.score_labels = {}

        # Reasoning explanations
        explanations_frame = ttk.LabelFrame(
            self.toxicity_frame, text="Reasoning Applied", padding="10")
        explanations_frame.grid(
            row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        explanations_frame.columnconfigure(0, weight=1)
        self.toxicity_frame.rowconfigure(2, weight=1)

        self.explanations_text = scrolledtext.ScrolledText(
            explanations_frame, height=8, wrap=tk.WORD)
        self.explanations_text.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        explanations_frame.rowconfigure(0, weight=1)

    def setup_paraphrases_tab(self):
        """Setup the paraphrases tab"""
        self.paraphrases_frame.columnconfigure(0, weight=1)
        self.paraphrases_frame.rowconfigure(0, weight=1)

        # Paraphrases list
        self.paraphrases_tree = ttk.Treeview(self.paraphrases_frame,
                                             columns=(
                                                 "rank", "text", "toxicity", "similarity", "fluency"),
                                             show="tree headings", height=15)

        # Configure columns
        self.paraphrases_tree.heading("#0", text="")
        self.paraphrases_tree.heading("rank", text="Rank")
        self.paraphrases_tree.heading("text", text="Paraphrase")
        self.paraphrases_tree.heading("toxicity", text="Toxicity")
        self.paraphrases_tree.heading("similarity", text="Similarity")
        self.paraphrases_tree.heading("fluency", text="Fluency")

        self.paraphrases_tree.column("#0", width=30)
        self.paraphrases_tree.column("rank", width=50)
        self.paraphrases_tree.column("text", width=400)
        self.paraphrases_tree.column("toxicity", width=80)
        self.paraphrases_tree.column("similarity", width=80)
        self.paraphrases_tree.column("fluency", width=80)

        self.paraphrases_tree.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar for treeview
        tree_scrollbar = ttk.Scrollbar(self.paraphrases_frame, orient=tk.VERTICAL,
                                       command=self.paraphrases_tree.yview)
        tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.paraphrases_tree.configure(yscrollcommand=tree_scrollbar.set)

    def setup_metadata_tab(self):
        """Setup the metadata tab - REMOVED"""
        pass

    def get_input_text(self) -> str:
        """Get text from input area"""
        text = self.text_input.get(1.0, tk.END).strip()
        if text == "Enter text here to analyze toxicity or generate paraphrases...":
            return ""
        return text

    def clear_input_placeholder(self, event=None):
        """Clear placeholder text when user clicks input"""
        current_text = self.text_input.get(1.0, tk.END).strip()
        if current_text == "Enter text here to analyze toxicity or generate paraphrases...":
            self.text_input.delete(1.0, tk.END)

    def validate_input(self) -> bool:
        """Validate input text"""
        text = self.get_input_text()
        if not text:
            messagebox.showwarning(
                "Warning", "Please enter some text to analyze.")
            return False
        if len(text) > 10000:
            messagebox.showwarning(
                "Warning", "Text is too long. Maximum 10,000 characters.")
            return False
        return True

    def set_processing_state(self, processing: bool):
        """Update UI state for processing"""
        self.is_processing = processing

        # Update button states
        state = tk.DISABLED if processing else tk.NORMAL
        self.analyze_btn.config(state=state)
        self.paraphrase_btn.config(state=state)

        # Update progress bar
        if processing:
            self.progress.start(10)
            self.status_label.config(
                text="Processing...", style="Processing.TLabel")
        else:
            self.progress.stop()

    def analyze_toxicity(self):
        """Analyze toxicity of input text"""
        if not self.validate_input() or self.is_processing:
            return

        text = self.get_input_text()

        # Run analysis in separate thread
        thread = threading.Thread(
            target=self._run_toxicity_analysis, args=(text,))
        thread.daemon = True
        thread.start()

    def _run_toxicity_analysis(self, text: str):
        """Run toxicity analysis in background thread"""
        try:
            self.root.after(0, lambda: self.set_processing_state(True))

            # Step 1: Get raw toxicity scores
            raw_scores = score_toxicity([text])[0]

            # Step 2: Apply reasoning
            reasoning_result = apply_reasoning(raw_scores)
            adjusted_scores = reasoning_result.get(
                'adjusted_labels', raw_scores)
            explanations = reasoning_result.get('explanations', [])

            # Update UI in main thread
            def update_ui():
                self._update_toxicity_results(
                    text, raw_scores, adjusted_scores, explanations)
                self.set_processing_state(False)

            self.root.after(0, update_ui)

        except Exception as e:
            def show_error():
                messagebox.showerror("Error", f"Analysis failed: {str(e)}")
                self.set_processing_state(False)
            self.root.after(0, show_error)

    def _update_toxicity_results(self, text: str, raw_scores: Dict,
                                 adjusted_scores: Dict, explanations: List):
        """Update toxicity results in UI"""
        # Update status
        toxicity_level = adjusted_scores.get('toxicity', 0.0)
        is_toxic = toxicity_level > 0.5

        if is_toxic:
            level_text = "HIGH" if toxicity_level > 0.7 else "MEDIUM"
            status_text = f"🚨 TOXIC ({level_text}): {toxicity_level:.1%}"
            style = "High.TLabel" if toxicity_level > 0.7 else "Medium.TLabel"
        else:
            status_text = f"✅ NON-TOXIC: {toxicity_level:.1%}"
            style = "Low.TLabel"

        self.status_label.config(text=status_text, style=style)

        # Update scores
        self._update_score_labels(raw_scores, adjusted_scores)

        # Update explanations
        self.explanations_text.delete(1.0, tk.END)
        if explanations:
            explanation_text = "\n".join([f"• {exp}" for exp in explanations])
            self.explanations_text.insert(
                tk.END, f"Reasoning applied:\n\n{explanation_text}")
        else:
            self.explanations_text.insert(
                tk.END, "No reasoning rules were applied to this text.")

        # Switch to toxicity tab
        self.results_notebook.select(0)

    def _update_score_labels(self, raw_scores: Dict, adjusted_scores: Dict):
        """Update toxicity score labels"""
        # Clear existing labels
        for widget in self.toxicity_frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame) and widget.cget("text") == "Toxicity Scores":
                scores_frame = widget
                break
        else:
            return

        # Clear frame
        for child in scores_frame.winfo_children():
            child.destroy()

        # Add headers
        ttk.Label(scores_frame, text="Category", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Label(scores_frame, text="Raw Score", font=("Arial", 10, "bold")).grid(
            row=0, column=1, sticky=tk.W, padx=(0, 20))
        ttk.Label(scores_frame, text="Adjusted", font=("Arial", 10, "bold")).grid(
            row=0, column=2, sticky=tk.W)

        # Add scores
        row = 1
        for label in ['toxicity', 'severe_toxicity', 'identity_attack', 'insult',
                      'threat', 'obscene', 'sexual_explicit']:
            display_name = label.replace('_', ' ').title()
            raw_score = raw_scores.get(label, 0.0)
            adj_score = adjusted_scores.get(label, 0.0)

            # Determine color based on score
            color = "red" if adj_score > 0.7 else "orange" if adj_score > 0.3 else "green"

            ttk.Label(scores_frame, text=display_name).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 20))
            ttk.Label(scores_frame, text=f"{raw_score:.1%}").grid(
                row=row, column=1, sticky=tk.W, padx=(0, 20))
            ttk.Label(scores_frame, text=f"{adj_score:.1%}", foreground=color).grid(
                row=row, column=2, sticky=tk.W)

            row += 1

    def paraphrase_text(self):
        """Paraphrase input text"""
        if not self.validate_input() or self.is_processing:
            return

        text = self.get_input_text()
        num_candidates = int(self.num_candidates.get())
        mode = self.mode.get()

        # Run paraphrasing in separate thread
        thread = threading.Thread(target=self._run_paraphrasing,
                                  args=(text, num_candidates, mode))
        thread.daemon = True
        thread.start()

    def _run_paraphrasing(self, text: str, num_candidates: int, mode: str):
        """Run paraphrasing in background thread"""
        try:
            self.root.after(0, lambda: self.set_processing_state(True))

            # Step 1: Analyze original text
            original_raw_toxicity = score_toxicity([text])[0]
            original_reasoning_result = apply_reasoning(original_raw_toxicity)
            original_adjusted_toxicity = original_reasoning_result.get(
                'adjusted_labels', original_raw_toxicity).get('toxicity', 0.0)

            # Check if fallback is needed
            if original_adjusted_toxicity <= 0.3:
                fallback_msg = "You don't need to reduce toxicity of this phrase"
                result = ParaphraseResult(
                    original=text,
                    candidates=[{
                        "text": fallback_msg,
                        "rank": 1,
                        "toxicity": 0.1,
                        "similarity": 0.0,
                        "fluency": 1.0
                    }],
                    metadata={"fallback_reason": "low_toxicity"},
                    fallback_used=True,
                    fallback_message=fallback_msg
                )

                def update_ui():
                    self._update_paraphrase_results(result)
                    self.set_processing_state(False)
                self.root.after(0, update_ui)
                return

            # Step 2: Generate paraphrases
            candidates = generate_paraphrases(
                text=text,
                num_return_sequences=num_candidates,
                mode=mode
            )

            if not candidates:
                fallback_msg = "You should reconsider saying something like that"
                result = ParaphraseResult(
                    original=text,
                    candidates=[{
                        "text": fallback_msg,
                        "rank": 1,
                        "toxicity": 0.1,
                        "similarity": 0.0,
                        "fluency": 1.0
                    }],
                    metadata={"fallback_reason": "no_candidates"},
                    fallback_used=True,
                    fallback_message=fallback_msg
                )

                def update_ui():
                    self._update_paraphrase_results(result)
                    self.set_processing_state(False)
                self.root.after(0, update_ui)
                return

            # Step 3: Score and rank candidates
            raw_toxicity_scores = score_toxicity(candidates)
            adjusted_results = [apply_reasoning(
                raw_scores) for raw_scores in raw_toxicity_scores]
            adjusted_toxicity_scores = [
                result.get('adjusted_labels', {}).get('toxicity', 0.0)
                for result in adjusted_results
            ]

            similarity_scores = taunt_equivalence_score(text, candidates)
            fluency_scores = score_fluency(candidates)

            # Rank candidates
            score_lists = {
                "toxicity": adjusted_toxicity_scores,
                "similarity": similarity_scores,
                "fluency": fluency_scores
            }
            ranking = custom_utility_score(score_lists)

            # Format results
            ranked_candidates = [
                {
                    "text": candidates[idx],
                    "rank": rank + 1,
                    "toxicity": adjusted_toxicity_scores[idx],
                    "similarity": similarity_scores[idx],
                    "fluency": fluency_scores[idx]
                }
                for rank, idx in enumerate(ranking)
            ]

            # Check if improvement is sufficient
            best_toxicity = ranked_candidates[0]["toxicity"]
            toxicity_improvement = original_adjusted_toxicity - best_toxicity

            if toxicity_improvement < 0.1 and original_adjusted_toxicity >= 0.7:
                fallback_msg = "You should reconsider saying something like that"
                result = ParaphraseResult(
                    original=text,
                    candidates=[{
                        "text": fallback_msg,
                        "rank": 1,
                        "toxicity": 0.1,
                        "similarity": 0.0,
                        "fluency": 1.0
                    }],
                    metadata={"fallback_reason": "insufficient_improvement"},
                    fallback_used=True,
                    fallback_message=fallback_msg
                )
            else:
                result = ParaphraseResult(
                    original=text,
                    candidates=ranked_candidates,
                    metadata={
                        "original_toxicity": original_adjusted_toxicity,
                        "best_candidate_toxicity": best_toxicity,
                        "toxicity_reduction": max(0.0, toxicity_improvement),
                        "candidates_generated": len(candidates),
                        "generation_mode": mode
                    }
                )

            def update_ui():
                self._update_paraphrase_results(result)
                self.set_processing_state(False)
            self.root.after(0, update_ui)

        except Exception as e:
            def show_error():
                messagebox.showerror("Error", f"Paraphrasing failed: {str(e)}")
                self.set_processing_state(False)
            self.root.after(0, show_error)

    def _update_paraphrase_results(self, result: ParaphraseResult):
        """Update paraphrase results in UI"""
        self.current_result = result

        # Clear existing results
        for item in self.paraphrases_tree.get_children():
            self.paraphrases_tree.delete(item)

        # Add new results
        for candidate in result.candidates:
            # Determine toxicity color
            tox_score = candidate["toxicity"]
            if tox_score > 0.7:
                tox_color = "red"
            elif tox_score > 0.3:
                tox_color = "orange"
            else:
                tox_color = "green"

            item = self.paraphrases_tree.insert("", "end", values=(
                candidate["rank"],
                candidate["text"],
                f"{candidate['toxicity']:.1%}",
                f"{candidate['similarity']:.1%}",
                f"{candidate['fluency']:.1%}"
            ))

            # Color toxicity column
            self.paraphrases_tree.set(
                item, "toxicity", f"{candidate['toxicity']:.1%}")

        # Switch to paraphrases tab
        self.results_notebook.select(1)

    def _update_metadata(self, result: ParaphraseResult):
        """Update metadata display - REMOVED"""
        pass

    def run(self):
        """Start the GUI application"""
        # Bind events
        self.text_input.bind("<FocusIn>", self.clear_input_placeholder)

        # Start the main loop
        self.root.mainloop()


if __name__ == "__main__":
    app = ToxicRadarGUI()
    app.run()
