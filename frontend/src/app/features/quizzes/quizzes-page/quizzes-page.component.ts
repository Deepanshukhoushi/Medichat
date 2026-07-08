import { ChangeDetectionStrategy, Component, HostListener, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { AsyncPipe, CommonModule } from '@angular/common';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { BehaviorSubject } from 'rxjs';

import { GlassCardComponent } from '../../../shared/components/glass-card/glass-card.component';
import { SectionHeadingComponent } from '../../../shared/components/section-heading/section-heading.component';
import { BackendApiService, QuizSession } from '../../../core/services/backend-api.service';
import { extractErrorMessage } from '../../../core/utils/extract-error-message';

interface QuizQuestion {
  question: string;
  options: string[];
  correct: number;
  explanation?: string;
}

interface QuizSessionDetail extends QuizSession {
  questions: QuizQuestion[];
}

@Component({
  selector: 'mc-quizzes-page',
  standalone: true,
  imports: [GlassCardComponent, SectionHeadingComponent, CommonModule, ReactiveFormsModule],
  templateUrl: './quizzes-page.component.html',
  styleUrl: './quizzes-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class QuizzesPageComponent implements OnInit, OnDestroy {
  private readonly api = inject(BackendApiService);
  private readonly fb = inject(FormBuilder);
  private timerId: number | null = null;

  sessions$ = new BehaviorSubject<QuizSession[]>([]);
  readonly isLoading = signal(false);
  readonly isGenerating = signal(false);
  readonly generateError = signal<string | null>(null);
  readonly quizLoadError = signal<string | null>(null);
  readonly timerSeconds = signal(0);

  form = this.fb.nonNullable.group({
    topic: ['', Validators.required],
    count: [5, [Validators.required, Validators.min(1), Validators.max(20)]]
  });

  readonly activeQuiz = signal<QuizSessionDetail | null>(null);
  readonly currentQuestionIndex = signal(0);
  readonly selectedAnswers = signal<Record<number, number>>({});
  readonly quizSubmitted = signal(false);
  readonly isSubmitting = signal(false);

  ngOnInit() {
    this.loadSessions(true);
  }

  ngOnDestroy(): void {
    this.stopTimer();
  }

  loadSessions(initial = false) {
    if (initial) this.isLoading.set(true);
    this.api.getQuizSessions().subscribe({
      next: (sessions) => {
        this.sessions$.next(sessions);
        this.isLoading.set(false);
      },
      error: (err) => {
        console.error('Failed to load sessions', err);
        this.isLoading.set(false);
      }
    });
  }

  generateQuiz() {
    if (this.form.invalid) return;
    this.isGenerating.set(true);
    this.generateError.set(null);
    const { topic, count } = this.form.getRawValue();

    this.api.generateQuiz(topic, count).subscribe({
      next: () => {
        this.isGenerating.set(false);
        this.form.reset({ topic: '', count: 5 });
        this.loadSessions();
      },
      error: (err) => {
        console.error('Failed to generate quiz', err);
        this.generateError.set(extractErrorMessage(err, 'Failed to generate quiz. Please try again.'));
        this.isGenerating.set(false);
      }
    });
  }

  takeQuiz(sessionId: string) {
    this.quizLoadError.set(null);
    this.api.getQuizSession(sessionId).subscribe({
      next: (session) => {
        this.activeQuiz.set(session as QuizSessionDetail);
        this.currentQuestionIndex.set(0);
        this.selectedAnswers.set({});
        this.quizSubmitted.set(session.score !== null);
        this.timerSeconds.set(0);
        this.startTimer();
      },
      error: (err) => {
        console.error('Failed to load quiz', err);
        this.quizLoadError.set('Failed to load quiz. Please try again.');
      }
    });
  }

  selectOption(optionIndex: number) {
    if (this.quizSubmitted()) return;
    this.selectedAnswers.update((answers) => ({ ...answers, [this.currentQuestionIndex()]: optionIndex }));
  }

  nextQuestion() {
    const quiz = this.activeQuiz();
    if (!quiz) return;
    this.currentQuestionIndex.update((i) => Math.min(i + 1, quiz.questions.length - 1));
  }

  prevQuestion() {
    this.currentQuestionIndex.update((i) => Math.max(i - 1, 0));
  }

  submitQuiz() {
    const quiz = this.activeQuiz();
    if (!quiz || !quiz.questions || this.isSubmitting()) return;

    const answers = quiz.questions.map((_, index) => this.selectedAnswers()[index] ?? -1);
    this.isSubmitting.set(true);
    this.api.submitQuizScore(quiz.id, answers).subscribe({
      next: (response) => {
        this.quizSubmitted.set(true);
        this.activeQuiz.set({ ...quiz, score: response.score });
        this.isSubmitting.set(false);
        this.stopTimer();
        this.loadSessions();
      },
      error: (err) => {
        console.error('Failed to submit quiz', err);
        this.isSubmitting.set(false);
      }
    });
  }

  closeQuiz() {
    this.activeQuiz.set(null);
    this.selectedAnswers.set({});
    this.quizSubmitted.set(false);
    this.currentQuestionIndex.set(0);
    this.timerSeconds.set(0);
    this.stopTimer();
  }

  isQuizComplete(): boolean {
    const quiz = this.activeQuiz();
    if (!quiz || !quiz.questions) return false;
    return Object.keys(this.selectedAnswers()).length === quiz.questions.length;
  }

  getCurrentQuestion(): QuizQuestion | null {
    const quiz = this.activeQuiz();
    if (!quiz) return null;
    return quiz.questions[this.currentQuestionIndex()] ?? null;
  }

  getElapsedTime(): string {
    const total = this.timerSeconds();
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }

  @HostListener('window:keydown', ['$event'])
  onKeyboardShortcut(event: KeyboardEvent): void {
    const quiz = this.activeQuiz();
    if (!quiz || this.quizSubmitted()) return;

    // Do not intercept keystrokes typed into form fields (e.g. a future
    // session-rename input). This prevents 'a' in a text field from
    // selecting answer option A.
    const target = event.target as Element;
    if (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      (target instanceof HTMLElement && target.isContentEditable)
    ) {
      return;
    }

    const key = event.key.toLowerCase();
    if (['a', 'b', 'c', 'd'].includes(key)) {
      event.preventDefault();
      this.selectOption(key.charCodeAt(0) - 97);
      return;
    }

    if (event.key === 'ArrowRight') {
      event.preventDefault();
      this.nextQuestion();
      return;
    }

    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      this.prevQuestion();
      return;
    }

    if (event.key === 'Enter' && this.isQuizComplete()) {
      event.preventDefault();
      this.submitQuiz();
    }
  }

  @HostListener('window:keydown.escape')
  onEscape(): void {
    if (this.activeQuiz()) {
      this.closeQuiz();
    }
  }

  restartQuiz(): void {
    const quiz = this.activeQuiz();
    if (!quiz) {
      return;
    }

    this.selectedAnswers.set({});
    this.quizSubmitted.set(false);
    this.currentQuestionIndex.set(0);
    this.timerSeconds.set(0);
    this.activeQuiz.set({ ...quiz, score: null });
    this.startTimer();
  }

  private startTimer(): void {
    this.stopTimer();
    this.timerId = window.setInterval(() => {
      this.timerSeconds.update((value) => value + 1);
    }, 1000);
  }

  private stopTimer(): void {
    if (this.timerId !== null) {
      window.clearInterval(this.timerId);
      this.timerId = null;
    }
  }
}
