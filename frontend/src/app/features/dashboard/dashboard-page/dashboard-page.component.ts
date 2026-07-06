import { ChangeDetectionStrategy, Component, inject, OnInit, signal } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
import { finalize } from 'rxjs';
import { Router, RouterLink } from '@angular/router';
import { LucideDynamicIcon } from '@lucide/angular';

import { GlassCardComponent } from '../../../shared/components/glass-card/glass-card.component';
import { SectionHeadingComponent } from '../../../shared/components/section-heading/section-heading.component';
import { BackendApiService, DashboardStats } from '../../../core/services/backend-api.service';
import { ChatService } from '../../../core/services/chat.service';
import { appIcons } from '../../../shared/icons/lucide-icons';

@Component({
  selector: 'mc-dashboard-page',
  standalone: true,
  imports: [CommonModule, DatePipe, RouterLink, LucideDynamicIcon, GlassCardComponent, SectionHeadingComponent],
  templateUrl: './dashboard-page.component.html',
  styleUrl: './dashboard-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class DashboardPageComponent implements OnInit {
  protected readonly icons = appIcons;
  private readonly api = inject(BackendApiService);
  private readonly chatService = inject(ChatService);
  private readonly router = inject(Router);

  dashboardStats = signal<DashboardStats | null>(null);
  isLoading = signal(true);
  errorMessage = signal<string | null>(null);

  ngOnInit() {
    this.loadDashboardStats();
  }

  protected retry(): void {
    this.loadDashboardStats();
  }

  private loadDashboardStats(): void {
    this.isLoading.set(true);
    this.api.getDashboardStats().pipe(
      finalize(() => this.isLoading.set(false))
    ).subscribe({
      next: (data) => {
        this.dashboardStats.set(data);
        this.errorMessage.set(null);
      },
      error: () => {
        this.errorMessage.set('Failed to load dashboard data. Please try again later.');
      }
    });
  }

  protected openConversation(id: string): void {
    this.chatService.loadHistory(id);
    this.router.navigate(['/app/chat']);
  }
}
