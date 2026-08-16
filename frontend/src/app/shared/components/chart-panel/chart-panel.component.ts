import { ChangeDetectionStrategy, Component, OnInit, effect, inject, input, signal } from '@angular/core';
import {
  ApexAxisChartSeries,
  ApexChart,
  ApexNonAxisChartSeries,
  ApexResponsive,
  ApexStroke,
  ApexXAxis,
  ChartComponent
} from 'ng-apexcharts';
import { ThemeService } from '../../../core/services/theme.service';

/** Read a resolved CSS custom property from the document root. */
function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

@Component({
  selector: 'mc-chart-panel',
  standalone: true,
  imports: [ChartComponent],
  templateUrl: './chart-panel.component.html',
  styleUrl: './chart-panel.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ChartPanelComponent {
  private readonly themeService = inject(ThemeService);

  readonly title = input.required<string>();
  readonly subtitle = input('');
  readonly mode = input<'line' | 'donut'>('line');
  readonly lineSeries = input<ApexAxisChartSeries>([]);
  readonly donutSeries = input<ApexNonAxisChartSeries>([]);
  readonly donutLabels = input<string[]>(['Mastered', 'Learning', 'Needs review']);

  // ── Theme-reactive color signals ─────────────────────────────────────────
  // Resolved at render time via getComputedStyle so they track dark/light mode.
  // An effect() re-resolves whenever themeService.activeTheme() changes.

  protected readonly lineColors = signal<string[]>([]);
  protected readonly donutColors = signal<string[]>([]);
  protected readonly lineGrid = signal<object>({});
  protected readonly lineChartOptions = signal<ApexChart>({
    type: 'area',
    toolbar: { show: false },
    background: 'transparent',
    foreColor: ''
  });
  protected readonly donutChartOptions = signal<ApexChart>({
    type: 'donut',
    background: 'transparent',
    height: 350,
    foreColor: ''
  });

  constructor() {
    // Re-resolve CSS variable values whenever the active theme changes.
    // This is the same pattern GlobeComponent uses for Three.js colors.
    effect(() => {
      // Reading activeTheme() registers this effect as a dependency.
      this.themeService.activeTheme();

      const accent  = cssVar('--color-accent');
      const success = cssVar('--color-success');
      const warning = cssVar('--color-warning');
      const secondary = cssVar('--color-secondary');

      // Build a grid border from the resolved line color at 40% opacity.
      // Can't use CSS variables directly in ApexCharts config, so we
      // construct an rgba string from the resolved hex/rgb value.
      const lineColor = cssVar('--color-line');
      const gridBorder = lineColor
        ? `color-mix(in srgb, ${lineColor} 40%, transparent)`
        : 'rgba(128,128,128,0.2)';

      this.lineColors.set([accent, warning]);
      this.donutColors.set([accent, success, warning]);
      this.lineGrid.set({ borderColor: gridBorder });
      this.lineChartOptions.set({
        type: 'area',
        toolbar: { show: false },
        background: 'transparent',
        foreColor: secondary
      });
      this.donutChartOptions.set({
        type: 'donut',
        background: 'transparent',
        height: 350,
        foreColor: secondary
      });
    });
  }

  protected readonly lineStroke: ApexStroke = {
    curve: 'smooth',
    width: 3
  };
  protected readonly lineXAxis: ApexXAxis = {
    categories: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  };
  protected readonly donutResponsive: ApexResponsive[] = [
    {
      breakpoint: 640,
      options: {
        chart: {
          width: 260
        }
      }
    }
  ];
}
