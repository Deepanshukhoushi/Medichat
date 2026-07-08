import { ChangeDetectionStrategy, Component, computed, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LucideDynamicIcon } from '@lucide/angular';
import { finalize } from 'rxjs';

import { GlassCardComponent } from '../../../shared/components/glass-card/glass-card.component';
import { SectionHeadingComponent } from '../../../shared/components/section-heading/section-heading.component';
import { BackendApiService, StudyStats } from '../../../core/services/backend-api.service';
import { appIcons } from '../../../shared/icons/lucide-icons';
import { ChartPanelComponent } from '../../../shared/components/chart-panel/chart-panel.component';

@Component({
  selector: 'mc-analytics-page',
  standalone: true,
  imports: [CommonModule, LucideDynamicIcon, GlassCardComponent, SectionHeadingComponent, ChartPanelComponent],
  templateUrl: './analytics-page.component.html',
  styleUrl: './analytics-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AnalyticsPageComponent implements OnInit {
  protected readonly icons = appIcons;
  protected readonly Math = Math;
  private readonly api = inject(BackendApiService);
  /**
   * Reference targets for the progress bars and donut chart.
   *
   * These are motivational monthly goals chosen to give a clear visual
   * sense of progress — not hard limits. The bars are labeled with these
   * values in the template (e.g. "2 / 50 sessions") so users always have
   * context for what 100% means.
   *
   *   total_sessions : 50  — ~12 sessions/week over a month
   *   total_messages : 500 — ~17 AI interactions/day
   *   unique_topics  : 20  — covering a broad topic spread each month
   *   streak_days    : 30  — a full calendar-month streak
   */
  protected readonly normalizationTargets = {
    total_sessions: 50,
    total_messages: 500,
    unique_topics: 20,
    streak_days: 30
  } as const;

  readonly stats = signal<StudyStats | null>(null);
  readonly errorMessage = signal<string | null>(null);
  readonly isLoading = signal(true);
  readonly chartLabels = ['Study sessions', 'AI messages', 'Topics explored', 'Streak days'];
  /** Subtitle shown below the donut chart so users know what 100% means. */
  readonly chartSubtitle = `Progress relative to monthly targets: ${this.normalizationTargets.total_sessions} sessions, ${this.normalizationTargets.unique_topics} topics, ${this.normalizationTargets.streak_days}-day streak.`;
  readonly chartSeries = computed(() => {
    const stats = this.stats();
    if (!stats) {
      return [];
    }

    return [
      this.toPercent(stats.total_sessions, this.normalizationTargets.total_sessions),
      this.toPercent(stats.total_messages, this.normalizationTargets.total_messages),
      this.toPercent(stats.unique_topics, this.normalizationTargets.unique_topics),
      this.toPercent(stats.streak_days, this.normalizationTargets.streak_days)
    ];
  });

  ngOnInit() {
    this.api.getStudyStats().pipe(
      finalize(() => this.isLoading.set(false))
    ).subscribe({
      next: (stats) => {
        this.stats.set(stats);
        this.errorMessage.set(null);
      },
      error: () => {
        this.errorMessage.set('Failed to load analytics data. Please try again later.');
      }
    });
  }

  private toPercent(value: number, target: number): number {
    return Number(Math.min((value / target) * 100, 100).toFixed(1));
  }
}
