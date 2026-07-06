import { ChangeDetectionStrategy, Component, computed, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LucideDynamicIcon } from '@lucide/angular';

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
  private readonly normalizationTargets = {
    total_sessions: 50,
    total_messages: 500,
    unique_topics: 20,
    streak_days: 30
  } as const;

  readonly stats = signal<StudyStats | null>(null);
  readonly errorMessage = signal<string | null>(null);
  readonly chartLabels = ['Study sessions', 'AI messages', 'Topics explored', 'Streak days'];
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
    this.api.getStudyStats().subscribe({
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
