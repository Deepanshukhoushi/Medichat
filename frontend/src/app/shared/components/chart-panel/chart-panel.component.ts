import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import {
  ApexAxisChartSeries,
  ApexChart,
  ApexNonAxisChartSeries,
  ApexResponsive,
  ApexStroke,
  ApexXAxis,
  NgApexchartsModule
} from 'ng-apexcharts';

@Component({
  selector: 'mc-chart-panel',
  standalone: true,
  imports: [NgApexchartsModule],
  templateUrl: './chart-panel.component.html',
  styleUrl: './chart-panel.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ChartPanelComponent {
  readonly title = input.required<string>();
  readonly subtitle = input('');
  readonly mode = input<'line' | 'donut'>('line');
  readonly lineSeries = input<ApexAxisChartSeries>([]);
  readonly donutSeries = input<ApexNonAxisChartSeries>([]);
  readonly donutLabels = input<string[]>(['Mastered', 'Learning', 'Needs review']);

  protected readonly lineChart: ApexChart = {
    type: 'area',
    toolbar: { show: false },
    background: 'transparent',
    foreColor: '#7a879f'
  };
  protected readonly lineStroke: ApexStroke = {
    curve: 'smooth',
    width: 3
  };
  protected readonly lineXAxis: ApexXAxis = {
    categories: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  };
  protected readonly donutChart: ApexChart = {
    type: 'donut',
    background: 'transparent'
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
