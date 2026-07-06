import { DOCUMENT } from '@angular/common';
import { computed, effect, inject, Injectable, signal } from '@angular/core';

import { ActiveTheme, ThemePreference } from '../../shared/models/theme.model';

const STORAGE_KEY = 'medichat-theme';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly document = inject(DOCUMENT);
  private readonly mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  private readonly preferenceSignal = signal<ThemePreference>(this.readStoredPreference());
  readonly preference = this.preferenceSignal.asReadonly();
  readonly activeTheme = computed<ActiveTheme>(() => {
    const preference = this.preferenceSignal();

    if (preference === 'system') {
      return this.mediaQuery.matches ? 'dark' : 'light';
    }

    return preference === 'dark' ? 'dark' : 'light';
  });

  constructor() {
    this.mediaQuery.addEventListener('change', () => {
      if (this.preferenceSignal() === 'system') {
        this.applyTheme();
      }
    });

    effect(() => {
      this.applyTheme();
    });
  }

  setPreference(preference: ThemePreference): void {
    this.preferenceSignal.set(preference);
    localStorage.setItem(STORAGE_KEY, preference);
  }

  cycleTheme(): void {
    const order: ThemePreference[] = ['light', 'dark', 'system'];
    const currentIndex = order.indexOf(this.preferenceSignal());
    this.setPreference(order[(currentIndex + 1) % order.length]);
  }

  private readStoredPreference(): ThemePreference {
    const storedTheme = localStorage.getItem(STORAGE_KEY);

    if (storedTheme === 'light' || storedTheme === 'dark' || storedTheme === 'system') {
      return storedTheme;
    }

    return 'system';
  }

  private applyTheme(): void {
    const root = this.document.documentElement;
    const activeTheme = this.activeTheme();

    root.setAttribute('data-theme', activeTheme);
    root.style.setProperty('view-transition-name', `theme-${activeTheme}`);
  }
}
