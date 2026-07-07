import { ChangeDetectionStrategy, Component, computed, HostListener, inject, signal, OnInit } from '@angular/core';
import { RouterLink, RouterOutlet, Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import { LucideDynamicIcon } from '@lucide/angular';

import { ChatService } from '../../core/services/chat.service';
import { AuthService } from '../../core/services/auth.service';
import { BackendApiService, UserProfile } from '../../core/services/backend-api.service';
import { NavigationService } from '../../core/services/navigation.service';
import { AppLogoComponent } from '../../shared/components/app-logo/app-logo.component';
import { ThemeToggleComponent } from '../../shared/components/theme-toggle/theme-toggle.component';
import { appIcons } from '../../shared/icons/lucide-icons';

@Component({
  selector: 'mc-app-shell-layout',
  standalone: true,
  imports: [RouterLink, RouterOutlet, FormsModule, LucideDynamicIcon, AppLogoComponent],
  templateUrl: './app-shell-layout.component.html',
  styleUrl: './app-shell-layout.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AppShellLayoutComponent implements OnInit {
  protected readonly icons = appIcons;
  protected readonly chatService = inject(ChatService);
  protected readonly authService = inject(AuthService);
  protected readonly backendApi = inject(BackendApiService);
  protected readonly navigationService = inject(NavigationService);
  protected readonly router = inject(Router);

  protected readonly isMobile = signal(typeof window !== 'undefined' ? window.innerWidth < 1024 : false);
  protected readonly isSidebarOpen = signal(!this.isMobile());
  protected readonly isContextPanelOpen = signal(false);
  protected readonly sidebarLabel = computed(() => (this.isSidebarOpen() ? 'Collapse sidebar' : 'Expand sidebar'));
  protected readonly isUserMenuOpen = signal(false);
  protected readonly userProfile = signal<UserProfile | null>(null);
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
    this.filteredHistory().slice(0, this.isSidebarOpen() ? 8 : 3)
  );

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

  protected deleteConversation(id: string, event?: MouseEvent): void {
    event?.stopPropagation();
    const confirmed = window.confirm('Delete this conversation? This cannot be undone.');
    if (!confirmed) {
      return;
    }

    this.backendApi.deleteConversation(id).subscribe({
      next: () => {
        if (this.chatService.activeConversationId() === id) {
          this.chatService.resetSession();
          this.router.navigate(['/app/chat']);
        }

        this.chatService.refreshConversationHistory().subscribe({
          error: () => undefined
        });
      },
      error: (err) => console.error('Failed to delete conversation', err)
    });
  }

  protected logout(): void {
    this.isUserMenuOpen.set(false);
    this.authService.logout().subscribe({
      next: () => this.router.navigate(['/auth/login']),
      error: () => this.router.navigate(['/auth/login'])
    });
  }

  ngOnInit() {
    // Load user profile — suppress 401 silently (guest/unauthenticated user)
    this.backendApi.getProfile().subscribe({
      next: (profile) => this.userProfile.set(profile),
      error: (err: HttpErrorResponse) => {
        if (err.status !== 401) {
          console.error('Failed to load profile for sidebar', err);
        }
        // 401 = guest / not logged in — stay silent, leave userProfile as null
      }
    });

    // Load conversation history from backend once and keep the chat service as the shared source of truth.
    this.isHistoryLoading.set(true);
    this.chatService.refreshConversationHistory().subscribe({
      next: () => this.isHistoryLoading.set(false),
      error: () => this.isHistoryLoading.set(false)
    });
  }
}
