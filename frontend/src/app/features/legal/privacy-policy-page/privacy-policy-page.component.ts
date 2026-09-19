import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'mc-privacy-policy-page',
  standalone: true,
  imports: [RouterLink],
  templateUrl: './privacy-policy-page.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class PrivacyPolicyPageComponent {
  readonly lastUpdated = 'September 2026';
}
