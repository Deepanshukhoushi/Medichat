import { Component } from '@angular/core';
import { animate, state, style, transition, trigger } from '@angular/animations';
import { Toast, ToastrService, ToastPackage } from 'ngx-toastr';
import { LucideDynamicIcon } from '@lucide/angular';
import { appIcons } from '../../icons/lucide-icons';

@Component({
  selector: '[mc-premium-toast]',
  standalone: true,
  imports: [LucideDynamicIcon],
  templateUrl: './premium-toast.component.html',
  animations: [
    trigger('flyInOut', [
      state('inactive', style({ opacity: 0, transform: 'translateY(6px) scale(0.98)' })),
      state('active', style({ opacity: 1, transform: 'translateY(0) scale(1)' })),
      state('removed', style({ opacity: 0, transform: 'translateY(-12px) scale(0.95)' })),
      transition('inactive => active', animate('180ms cubic-bezier(0.175, 0.885, 0.32, 1.275)')),
      transition('active => removed', animate('140ms ease-out')),
    ]),
  ],
  host: {
    'class': 'mc-premium-toast-host',
    '[@flyInOut]': 'state()',
    '(mouseenter)': 'stickAround()',
    '(mouseleave)': 'delayedHideToast()',
    '(click)': 'tapToast()',
    'style': 'display: block; margin-bottom: 12px; pointer-events: auto;'
  },
  preserveWhitespaces: false,
})
export class PremiumToastComponent extends Toast {
  protected readonly icons = appIcons;
  
  // Expose width as a percentage string for the progress bar
  progressWidth = () => this.width() + '%';
}
