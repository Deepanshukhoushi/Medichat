import {
  ChangeDetectionStrategy,
  Component,
  OnInit,
  signal
} from '@angular/core';
import { RouterLink } from '@angular/router';

const CONSENT_KEY = 'medichat-cookie-consent';

@Component({
  selector: 'mc-cookie-consent',
  standalone: true,
  imports: [RouterLink],
  templateUrl: './cookie-consent.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class CookieConsentComponent implements OnInit {
  /** Whether the banner should be visible */
  readonly visible = signal(false);

  ngOnInit(): void {
    // Only show if the user has not already responded
    if (!localStorage.getItem(CONSENT_KEY)) {
      // Small delay so it doesn't flash immediately on page load
      setTimeout(() => this.visible.set(true), 1200);
    }
  }

  accept(): void {
    localStorage.setItem(CONSENT_KEY, 'accepted');
    this.visible.set(false);
  }

  decline(): void {
    localStorage.setItem(CONSENT_KEY, 'declined');
    this.visible.set(false);
  }
}
