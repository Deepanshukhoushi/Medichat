import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'mc-terms-page',
  standalone: true,
  imports: [RouterLink],
  templateUrl: './terms-page.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class TermsPageComponent {
  readonly lastUpdated = 'September 2026';
}
