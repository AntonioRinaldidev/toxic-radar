'use client';
import { useState, useCallback } from 'react';
import '@/styles/Home.css';

// Enhanced interfaces with better typing
interface ToxicityClassification {
	toxicity: number;
	severe_toxicity: number;
	obscene: number;
	identity_attack: number;
	insult: number;
	threat: number;
	sexual_explicit: number;
}

interface ToxicitySummary {
	is_toxic: boolean;
	toxicity_level: 'high' | 'medium' | 'low';
	main_issues: string[];
	reasoning_applied: boolean;
	toxicity_adjustment: number;
}

interface ToxicityAnalysis {
	raw_classification: ToxicityClassification;
	adjusted_classification: ToxicityClassification;
	reasoning_explanations: string[];
	summary: ToxicitySummary;
	text: string;
}

interface ParaphraseCandidate {
	text: string;
	toxicity: number;
	similarity: number;
	fluency: number;
	rank: number;
}

interface ParaphraseResult {
	original: string;
	candidates: ParaphraseCandidate[];
	metadata: {
		original_toxicity: number;
		best_candidate_toxicity: number;
		toxicity_reduction: number;
		candidates_generated: number;
	};
}

type ProcessingMode = 'analyze' | 'paraphrase' | 'idle';

export default function ToxicRadarPage() {
	// State management
	const [inputText, setInputText] = useState('');
	const [toxicityAnalysis, setToxicityAnalysis] =
		useState<ToxicityAnalysis | null>(null);
	const [paraphraseResult, setParaphraseResult] =
		useState<ParaphraseResult | null>(null);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string>('');
	const [processingMode, setProcessingMode] = useState<ProcessingMode>('idle');
	const [numCandidates, setNumCandidates] = useState(3);

	// Clear results and errors
	const clearResults = useCallback(() => {
		setToxicityAnalysis(null);
		setParaphraseResult(null);
		setError('');
	}, []);

	// Enhanced error handling
	const handleApiError = useCallback((error: any, defaultMessage: string) => {
		if (error?.detail) {
			setError(error.detail);
		} else if (error?.message) {
			setError(error.message);
		} else {
			setError(defaultMessage);
		}
	}, []);

	// Toxicity analysis handler
	const handleAnalyzeToxicity = useCallback(async () => {
		if (!inputText.trim()) {
			setError('Please enter some text to analyze.');
			return;
		}

		setIsLoading(true);
		setProcessingMode('analyze');
		clearResults();

		try {
			const response = await fetch('http://localhost:8000/analyze', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ text: inputText.trim() }),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw errorData;
			}

			const data = await response.json();
			setToxicityAnalysis({
				raw_classification: data.analysis.raw_classification,
				adjusted_classification: data.analysis.adjusted_classification,
				reasoning_explanations: data.analysis.reasoning_explanations || [],
				summary: data.analysis.summary,
				text: data.text,
			});
		} catch (err) {
			handleApiError(err, 'Failed to analyze toxicity. Please try again.');
		} finally {
			setIsLoading(false);
			setProcessingMode('idle');
		}
	}, [inputText, clearResults, handleApiError]);

	// Paraphrasing handler
	const handleParaphrase = useCallback(async () => {
		if (!inputText.trim()) {
			setError('Please enter some text to paraphrase.');
			return;
		}

		setIsLoading(true);
		setProcessingMode('paraphrase');
		clearResults();

		try {
			const response = await fetch('http://localhost:8000/paraphrase', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					text: inputText.trim(),
					num_candidates: numCandidates,
					mode: 'auto',
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw errorData;
			}

			const data = await response.json();
			setParaphraseResult({
				original: data.original,
				candidates: data.candidates,
				metadata: data.metadata,
			});
		} catch (err) {
			handleApiError(err, 'Failed to generate paraphrases. Please try again.');
		} finally {
			setIsLoading(false);
			setProcessingMode('idle');
		}
	}, [
		inputText,
		numCandidates,
		clearResults,
		handleApiError,
		handleAnalyzeToxicity,
	]);

	// Utility functions
	const formatScore = useCallback((score: number): string => {
		if (typeof score !== 'number' || isNaN(score)) return 'N/A';
		return `${(score * 100).toFixed(1)}%`;
	}, []);

	const getToxicityClass = useCallback((score: number): string => {
		if (typeof score !== 'number' || isNaN(score)) return 'unknown';
		if (score >= 0.7) return 'high';
		if (score >= 0.3) return 'medium';
		return 'low';
	}, []);

	const formatLabel = useCallback((label: string): string => {
		return label
			.split('_')
			.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
			.join(' ');
	}, []);

	const getProcessingMessage = useCallback(() => {
		switch (processingMode) {
			case 'analyze':
				return 'Analyzing toxicity...';
			case 'paraphrase':
				return 'Generating paraphrases...';
			default:
				return 'Processing...';
		}
	}, [processingMode]);

	return (
		<div className='toxic-radar-container'>
			{/* Hero Section */}
			<header className='hero-section'>
				<div className='hero-content'>
					<h1 className='hero-title'>
						<span className='radar-icon'></span>
						ToxicRadar
					</h1>
					<p className='hero-subtitle'>
						AI-Powered Content Analysis & Intelligent Paraphrasing
					</p>
					<div className='hero-features'>
						<span className='feature-tag'>Soft-CSP Reasoning</span>
						<span className='feature-tag'>Voting Theory</span>
						<span className='feature-tag'>Multi-Agent System</span>
					</div>
				</div>
			</header>

			{/* Main Content */}
			<main className='main-content'>
				{/* Input Section */}
				<section className='input-section'>
					<div className='input-header'>
						<h2>Text Analysis</h2>
						<p>
							Enter text below to analyze toxicity levels or generate safer
							alternatives
						</p>
					</div>

					<div className='input-container'>
						<textarea
							className='text-input'
							placeholder='Type or paste your text here for analysis...'
							value={inputText}
							onChange={(e) => setInputText(e.target.value)}
							rows={6}
							maxLength={500}
							disabled={isLoading}
						/>
						<div className='input-footer'>
							<span className='char-counter'>
								{inputText.length}/500 characters
							</span>
						</div>
					</div>

					<div className='action-buttons'>
						<button
							className='btn btn-primary'
							onClick={handleAnalyzeToxicity}
							disabled={isLoading || !inputText.trim()}>
							{isLoading && processingMode === 'analyze' ? (
								<>
									<span className='loading-spinner'></span>
									Analyzing...
								</>
							) : (
								<>
									<span className='btn-icon'></span>
									Analyze Toxicity
								</>
							)}
						</button>

						<button
							className='btn btn-secondary'
							onClick={handleParaphrase}
							disabled={isLoading || !inputText.trim()}>
							{isLoading && processingMode === 'paraphrase' ? (
								<>
									<span className='loading-spinner'></span>
									Paraphrasing...
								</>
							) : (
								<>
									<span className='btn-icon'></span>
									Generate Paraphrases
								</>
							)}
						</button>
					</div>
				</section>

				{/* Status Messages */}
				{isLoading && (
					<div className='status-message loading'>
						<div className='status-content'>
							<span className='loading-spinner large'></span>
							<span>{getProcessingMessage()}</span>
						</div>
					</div>
				)}

				{error && (
					<div className='status-message error'>
						<span className='status-icon'>⚠️</span>
						<span>{error}</span>
						<button onClick={() => setError('')} className='close-btn'>
							×
						</button>
					</div>
				)}

				{/* Results Sections */}
				{toxicityAnalysis && (
					<section className='results-section toxicity-results'>
						<div className='section-header'>
							<h2>Toxicity Analysis Results</h2>
							<div
								className={`overall-status ${
									toxicityAnalysis.summary.is_toxic ? 'toxic' : 'clean'
								}`}>
								<span className='status-icon'>
									{toxicityAnalysis.summary.is_toxic ? '🚨' : '✅'}
								</span>
								<div className='status-details'>
									<span className='status-text'>
										{toxicityAnalysis.summary.is_toxic
											? `Toxic Content Detected`
											: 'Content is Clean'}
									</span>
									<span className='status-level'>
										Level:{' '}
										{toxicityAnalysis.summary.toxicity_level.toUpperCase()}
									</span>
								</div>
							</div>
						</div>

						{/* Original Text Display */}
						<div className='analyzed-text-display'>
							<h3>Analyzed Text</h3>
							<div className='text-content'>"{toxicityAnalysis.text}"</div>
						</div>

						{/* Main Issues */}
						{toxicityAnalysis.summary.main_issues?.length > 0 && (
							<div className='main-issues-display'>
								<h4>Identified Issues:</h4>
								<div className='issues-tags'>
									{toxicityAnalysis.summary.main_issues.map((issue, index) => (
										<span key={index} className='issue-tag'>
											{formatLabel(issue)}
										</span>
									))}
								</div>
							</div>
						)}

						{/* Toxicity Scores Grid */}
						<div className='scores-section'>
							<h3>Toxicity Scores</h3>
							<div className='scores-grid'>
								{Object.entries(toxicityAnalysis.adjusted_classification).map(
									([label, score]) => (
										<div key={label} className='score-card'>
											<div className='score-header'>
												<span className='score-label'>
													{formatLabel(label)}
												</span>
											</div>
											<div className={`score-value ${getToxicityClass(score)}`}>
												{formatScore(score)}
											</div>
											<div className='score-bar'>
												<div
													className={`score-fill ${getToxicityClass(score)}`}
													style={{ width: `${score * 100}%` }}></div>
											</div>
										</div>
									),
								)}
							</div>
						</div>

						{/* Raw vs Adjusted Comparison */}
						<details className='comparison-section'>
							<summary>View Raw vs Adjusted Scores Comparison</summary>
							<div className='comparison-content'>
								<div className='comparison-side'>
									<h4>Raw Classification</h4>
									<div className='mini-scores-grid'>
										{Object.entries(toxicityAnalysis.raw_classification).map(
											([label, score]) => (
												<div key={label} className='mini-score-item'>
													<span className='mini-label'>
														{formatLabel(label)}
													</span>
													<span
														className={`mini-value ${getToxicityClass(score)}`}>
														{formatScore(score)}
													</span>
												</div>
											),
										)}
									</div>
								</div>
								<div className='comparison-side'>
									<h4>After AI Reasoning</h4>
									<div className='mini-scores-grid'>
										{Object.entries(
											toxicityAnalysis.adjusted_classification,
										).map(([label, score]) => (
											<div key={label} className='mini-score-item'>
												<span className='mini-label'>{formatLabel(label)}</span>
												{(() => {
													const raw =
														toxicityAnalysis.raw_classification[
															label as keyof ToxicityClassification
														];
													const adjusted =
														toxicityAnalysis.adjusted_classification[
															label as keyof ToxicityClassification
														];
													if (adjusted > raw) {
														return (
															<span className='adjustment-indicator increased'>
																↑ Increased
															</span>
														);
													} else if (adjusted < raw) {
														return (
															<span className='adjustment-indicator decreased'>
																↓ Decreased
															</span>
														);
													} else {
														return null;
													}
												})()}
												<span
													className={`mini-value ${getToxicityClass(score)}`}>
													{formatScore(score)}
												</span>
											</div>
										))}
									</div>
								</div>
							</div>
						</details>
					</section>
				)}

				{/* Paraphrase Results */}
				{paraphraseResult && (
					<section className='results-section paraphrase-results'>
						<div className='section-header'>
							<h2>Generated Paraphrases</h2>
							<div className='paraphrase-stats'>
								<div className='stat-item'>
									<span className='stat-label'>Original Toxicity:</span>
									<span
										className={`stat-value ${getToxicityClass(
											paraphraseResult.metadata.original_toxicity,
										)}`}>
										{formatScore(paraphraseResult.metadata.original_toxicity)}
									</span>
								</div>
								<div className='stat-item'>
									<span className='stat-label'>Best Alternative:</span>
									<span
										className={`stat-value ${getToxicityClass(
											paraphraseResult.metadata.best_candidate_toxicity,
										)}`}>
										{formatScore(
											paraphraseResult.metadata.best_candidate_toxicity,
										)}
									</span>
								</div>
								<div className='stat-item improvement'>
									<span className='stat-label'>Toxicity Reduction:</span>
									<span className='stat-value'>
										-{formatScore(paraphraseResult.metadata.toxicity_reduction)}
									</span>
								</div>
							</div>
						</div>

						<div className='paraphrase-list'>
							{paraphraseResult.candidates.map((candidate, index) => (
								<div key={index} className='paraphrase-card'>
									<div className='paraphrase-header'>
										<div className='rank-badge'>#{candidate.rank}</div>
										<div className='candidate-scores'>
											<span
												className={`score-chip toxicity ${getToxicityClass(
													candidate.toxicity,
												)}`}>
												Toxicity: {formatScore(candidate.toxicity)}
											</span>
											<span className='score-chip similarity'>
												Similarity: {formatScore(candidate.similarity)}
											</span>
											<span className='score-chip fluency'>
												Fluency: {formatScore(candidate.fluency)}
											</span>
										</div>
									</div>
									<div className='paraphrase-text'>{candidate.text}</div>
									<div className='paraphrase-footer'>
										<button
											className='copy-btn'
											onClick={() =>
												navigator.clipboard.writeText(candidate.text)
											}>
											Copy
										</button>
									</div>
								</div>
							))}
						</div>
					</section>
				)}
			</main>

			{/* Footer */}
			<footer className='app-footer'>
				<div className='footer-content'>
					<p>
						<strong>ToxicRadar</strong> - Intelligent toxicity detection using
						Soft-CSP reasoning, voting theory, and multi-agent systems
					</p>
					<div className='footer-links'>
						<a href='#' className='footer-link'>
							GitHub
						</a>
						<a href='#' className='footer-link'>
							Documentation
						</a>
						<a href='#' className='footer-link'>
							API Reference
						</a>
					</div>
				</div>
			</footer>
		</div>
	);
}
