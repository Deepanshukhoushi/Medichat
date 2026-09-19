import { ChangeDetectionStrategy, Component, computed, HostListener, inject, signal, OnInit } from '@angular/core';
import { RouterLink, RouterOutlet, Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { LucideDynamicIcon } from '@lucide/angular';

import { ConversationSummary } from '../../shared/models/chat.model';

import { ChatService } from '../../core/services/chat.service';
import { AuthService } from '../../core/services/auth.service';
import { BackendApiService } from '../../core/services/backend-api.service';
import { ProfileService } from '../../core/services/profile.service';
import { NavigationService } from '../../core/services/navigation.service';
import { LogoComponent } from '../../shared/components/logo/logo.component';

import { appIcons } from '../../shared/icons/lucide-icons';

@Component({
  selector: 'mc-app-shell-layout',
  standalone: true,
  imports: [RouterLink, RouterOutlet, FormsModule, LucideDynamicIcon, LogoComponent],
  templateUrl: './app-shell-layout.component.html',
  styleUrl: './app-shell-layout.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AppShellLayoutComponent implements OnInit {
  protected readonly icons = appIcons;
  protected readonly chatService = inject(ChatService);
  protected readonly authService = inject(AuthService);
  protected readonly backendApi = inject(BackendApiService);
  protected readonly profileService = inject(ProfileService);
  protected readonly navigationService = inject(NavigationService);
  protected readonly router = inject(Router);

  protected readonly isMobile = signal(typeof window !== 'undefined' ? window.innerWidth < 1024 : false);
  protected readonly isSidebarOpen = signal(!this.isMobile());
  protected readonly isContextPanelOpen = signal(false);
  protected readonly sidebarLabel = computed(() => (this.isSidebarOpen() ? 'Collapse sidebar' : 'Expand sidebar'));
  protected readonly isUserMenuOpen = signal(false);
  /** Alias to the shared ProfileService signal — no extra network call needed. */
  protected readonly userProfile = this.profileService.profile;
  protected readonly isHistoryLoading = signal(false);
  protected readonly searchTerm = signal('');
  protected readonly isGuestUser = computed(() => {
    const profile = this.userProfile();
    // null profile means unauthenticated (401) → treat as guest
    if (profile === null) return true;
    return profile.user_id.startsWith('guest_');
  });

  protected readonly filteredHistory = computed(() =>
    this.chatService.conversationHistory().filter((c) =>
      c.title.toLowerCase().includes(this.searchTerm().toLowerCase())
    )
  );

  protected readonly navItems = this.navigationService.navItems;
  protected readonly userNavItems = this.navigationService.userNavItems;
  protected readonly historyPreview = computed(() =>
    this.isSidebarOpen() ? this.filteredHistory() : this.filteredHistory().slice(0, 3)
  );

  /** Groups filtered history into date buckets for issue #16 (flat unorganised list). */
  protected readonly groupedHistory = computed(() => {
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterdayStart = new Date(todayStart.getTime() - 86_400_000);
    const weekStart = new Date(todayStart.getTime() - 7 * 86_400_000);

    const buckets: { label: string; items: ConversationSummary[] }[] = [
      { label: 'Today', items: [] },
      { label: 'Yesterday', items: [] },
      { label: 'This Week', items: [] },
      { label: 'Older', items: [] },
    ];

    for (const c of this.filteredHistory()) {
      const d = c.updatedAt ? new Date(c.updatedAt) : null;
      if (!d || isNaN(d.getTime())) {
        buckets[3].items.push(c);
        continue;
      }
      const day = new Date(d.getFullYear(), d.getMonth(), d.getDate());
      if (day >= todayStart)           buckets[0].items.push(c);
      else if (day >= yesterdayStart)  buckets[1].items.push(c);
      else if (day >= weekStart)       buckets[2].items.push(c);
      else                             buckets[3].items.push(c);
    }

    return buckets.filter(b => b.items.length > 0);
  });

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    if (!target.closest('.user-menu-container')) {
      this.isUserMenuOpen.set(false);
    }
  }

  @HostListener('window:resize')
  onResize(): void {
    if (typeof window !== 'undefined') {
      const mobile = window.innerWidth < 1024;
      this.isMobile.set(mobile);
      if (mobile && this.isSidebarOpen()) {
        this.isSidebarOpen.set(false);
      } else if (!mobile && !this.isSidebarOpen()) {
        this.isSidebarOpen.set(true);
      }
    }
  }

  protected toggleSidebar(): void {
    this.isSidebarOpen.update((value) => !value);
  }

  protected toggleContextPanel(): void {
    this.isContextPanelOpen.update((value) => !value);
  }

  protected toggleUserMenu(): void {
    this.isUserMenuOpen.update((value) => !value);
  }

  protected closeUserMenu(): void {
    this.isUserMenuOpen.set(false);
  }

  protected isActiveRoute(route: string): boolean {
    const currentUrl = this.router.url.split('?')[0].split('#')[0];
    return currentUrl === route || currentUrl.startsWith(route + '/');
  }

  protected startNewChat(): void {
    this.chatService.resetSession();
    this.router.navigate(['/app/chat']);
  }

  protected loadConversation(id: string): void {
    this.chatService.loadHistory(id);
    this.router.navigate(['/app/chat']);
  }

  protected readonly deletingConversationId = signal<string | null>(null);
  protected readonly isDeleting = signal(false);

  protected requestDelete(id: string, event?: MouseEvent): void {
    event?.stopPropagation();
    this.deletingConversationId.set(id);
  }

  protected cancelDelete(): void {
    this.deletingConversationId.set(null);
  }

  protected confirmDelete(): void {
    const id = this.deletingConversationId();
    if (!id) return;

    this.isDeleting.set(true);
    this.backendApi.deleteConversation(id).subscribe({
      next: () => {
        if (this.chatService.activeConversationId() === id) {
          this.chatService.resetSession();
          this.router.navigate(['/app/chat']);
        }

        this.chatService.refreshConversationHistory().subscribe({
          next: () => {
            this.isDeleting.set(false);
            this.deletingConversationId.set(null);
          },
          error: () => {
            this.isDeleting.set(false);
            this.deletingConversationId.set(null);
          }
        });
      },
      error: (err) => {
        console.error('Failed to delete conversation', err);
        this.isDeleting.set(false);
        this.deletingConversationId.set(null);
      }
    });
  }

  protected logout(): void {
    this.isUserMenuOpen.set(false);
    this.authService.logout().subscribe({
      next: () => {
        this.profileService.invalidate();
        this.router.navigate(['/auth/login']);
      },
      error: () => {
        this.profileService.invalidate();
        this.router.navigate(['/auth/login']);
      }
    });
  }

  ngOnInit() {
    // Profile is populated by ProfileService (cached — no extra network call here).
    // Fire the observable so the signal gets set if it hasn't been yet.
    this.profileService.profile$.subscribe({ error: () => undefined });

    // Load conversation history from backend once and keep the chat service as the shared source of truth.
    this.isHistoryLoading.set(true);
    this.chatService.refreshConversationHistory().subscribe({
      next: () => this.isHistoryLoading.set(false),
      error: () => this.isHistoryLoading.set(false)
    });
  }
}
