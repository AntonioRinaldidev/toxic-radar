'use client';
import { useState } from 'react';
import '@/styles/Home.css';

interface ToxicityScore {
	raw_classification: {
		toxicity: number;
		severe_toxicity: number;
		obscene: number;
		identity_attack: number;
		insult: number;
		threat: number;
		sexual_explicit: number;
	};

	adjusted_classification: {
		toxicity: number;
		severe_toxicity: number;
		obscene: number;
		identity_attack: number;
		insult: number;
		threat: number;
		sexual_explicit: number;
	};

	summary: {
		is_toxic: boolean;
		toxicity_level: string;
		main_issues: string[];
	};

	text: string;
}

export default function Home() {
	const [inputText, setInputText] = useState('');
	const [paraphrasedText, setParaphrasedText] = useState([]);
	const [toxicityScores, setToxicityScores] = useState<ToxicityScore | null>(
		null,
	);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string>('');

	const handleParaphrase = async () => {
		if (!inputText.trim()) {
			setError('Please enter some text to paraphrase.');
			return;
		}
		setIsLoading(true);
		setError('');
		setParaphrasedText([]);
		setToxicityScores(null);

		try {
			const response = await fetch('http://localhost:8000/paraphrase', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					text: inputText,
					num_candidates: 3,
					mode: 'auto',
				}),
			});

			if (!response.ok) {
				const errorData = await response.json();
				console.log('errorData', errorData);
				throw new Error(errorData.detail || 'Failed to paraphrase text.');
			}

			const data = await response.json();
			const paraphrases = data.candidates.map(
				(candidate: any) => candidate.text,
			);
			setParaphrasedText(paraphrases);
		} catch (err) {
			setError(
				err instanceof Error ? err.message : 'An unexpected error occurred.',
			);
		} finally {
			setIsLoading(false);
		}
	};

	const handleClassifyToxicity = async () => {
		if (!inputText.trim()) {
			setError('Please enter some text to classify toxicity.');
			return;
		}

		setIsLoading(true);
		setError('');
		setToxicityScores(null);

		try {
			const response = await fetch('http://localhost:8000/analyze', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({ text: inputText }),
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(errorData.detail || 'Failed to classify toxicity.');
			}

			const data = await response.json();

			setToxicityScores({
				raw_classification: data.analysis.raw_classification,
				adjusted_classification: data.analysis.adjusted_classification,
				summary: data.analysis.summary,
				text: data.text,
			});
		} catch (err: any) {
			setError(err.message);
		} finally {
			setIsLoading(false);
		}
	};

	// Helper functions
	const formatScore = (score: number): string => {
		if (typeof score !== 'number' || isNaN(score)) return 'N/A';
		return `${(score * 100).toFixed(1)}%`;
	};

	const getToxicityClass = (score: number): string => {
		if (typeof score !== 'number' || isNaN(score)) return 'unknown';
		if (score >= 0.7) return 'high';
		if (score >= 0.3) return 'medium';
		return 'low';
	};

	const formatLabel = (label: string): string => {
		return label
			.split('_')
			.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
			.join(' ');
	};

	return (
		<div className='page-Container'>
			<div className='page-Main'>
				<h1 className='title'>Welcome to TOXIC RADAR</h1>
				<p className='description'>
					Your one-stop platform for toxic online content detection and
					paraphrasing
				</p>

				{/* Input Section */}
				<div className='input-section'>
					<textarea
						className='text-input'
						placeholder='Enter text here to detect toxicity or paraphrase...'
						value={inputText}
						onChange={(e) => setInputText(e.target.value)}
						rows={8}></textarea>
					<div className='button-group'>
						<button
							onClick={handleParaphrase}
							disabled={isLoading || !inputText.trim()}>
							{isLoading ? 'Processing...' : 'Paraphrase'}
						</button>
						<button
							onClick={handleClassifyToxicity}
							disabled={isLoading || !inputText.trim()}
							className='button-secondary'>
							Analyze Toxicity Only
						</button>
					</div>
				</div>

				{/* Status Messages */}
				{isLoading && <p className='message-status'>Loading...</p>}
				{error && <p className='message-error'>{error}</p>}

				{/* Analysis Results */}
				{toxicityScores && (
					<div className='toxicity-results'>
						{/* Summary Section */}
						<div className='analysis-summary'>
							<h2>Toxicity Analysis Results</h2>

							{/* Original Text - moved to top */}
							<div className='analyzed-text'>
								<h4>Analyzed Text:</h4>
								<p className='text-content'>"{toxicityScores.text}"</p>
							</div>

							{/* Overall Status */}
							<div
								className={`overall-status ${
									toxicityScores.summary.is_toxic ? 'toxic' : 'clean'
								}`}>
								<span className='status-icon'>
									{toxicityScores.summary.is_toxic ? '🚨' : '✅'}
								</span>
								<span className='status-text'>
									{toxicityScores.summary.is_toxic
										? `Toxic Content (${toxicityScores.summary.toxicity_level})`
										: 'Clean Content'}
								</span>
							</div>

							{/* Main Issues */}
							{toxicityScores.summary.main_issues?.length > 0 && (
								<div className='main-issues'>
									<strong>Main Issues:</strong>{' '}
									{toxicityScores.summary.main_issues.join(', ')}
								</div>
							)}
						</div>

						{/* Classification Scores */}
						<div className='classification-section'>
							<h3>Toxicity Scores</h3>
							<div className='score-grid'>
								{Object.entries(toxicityScores.adjusted_classification).map(
									([label, score]) => (
										<div key={label} className='score-item'>
											<span className='score-label'>{formatLabel(label)}</span>
											<span
												className={`score-value ${getToxicityClass(score)}`}>
												{formatScore(score)}
											</span>
										</div>
									),
								)}
							</div>
						</div>

						{/* Raw vs Adjusted Comparison */}
						<details className='comparison-section'>
							<summary>View Raw vs Adjusted Scores</summary>
							<div className='comparison-grid'>
								<div className='raw-scores'>
									<h4>Raw Classification</h4>
									<div className='score-grid'>
										{Object.entries(toxicityScores.raw_classification).map(
											([label, score]) => (
												<div key={label} className='score-item'>
													<span className='score-label'>
														{formatLabel(label)}
													</span>
													<span
														className={`score-value ${getToxicityClass(
															score,
														)}`}>
														{formatScore(score)}
													</span>
												</div>
											),
										)}
									</div>
								</div>

								<div className='adjusted-scores'>
									<h4>After Reasoning</h4>
									<div className='score-grid'>
										{Object.entries(toxicityScores.adjusted_classification).map(
											([label, score]) => (
												<div key={label} className='score-item'>
													<span className='score-label'>
														{formatLabel(label)}
													</span>
													<span
														className={`score-value ${getToxicityClass(
															score,
														)}`}>
														{formatScore(score)}
													</span>
													{/* Show difference if there's a change */}
													{toxicityScores.raw_classification[
														label as keyof typeof toxicityScores.raw_classification
													] !== score && (
														<span className='score-change'>(adjusted)</span>
													)}
												</div>
											),
										)}
									</div>
								</div>
							</div>
						</details>
					</div>
				)}

				{/* Paraphrase Results */}
				{paraphrasedText.length > 0 && (
					<div className='paraphrase-results'>
						<h2>Paraphrased Suggestions</h2>
						<ul className='paraphrase-list'>
							{paraphrasedText.map((pText, index) => (
								<li key={index} className='paraphrase-item'>
									{pText}
								</li>
							))}
						</ul>
					</div>
				)}
			</div>
		</div>
	);
}
