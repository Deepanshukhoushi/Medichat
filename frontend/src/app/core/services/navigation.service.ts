import { Injectable } from '@angular/core';

import { appIcons } from '../../shared/icons/lucide-icons';
import { NavItem } from '../../shared/models/navigation.model';

@Injectable({ providedIn: 'root' })
export class NavigationService {
  readonly navItems: NavItem[] = [
    {
      label: 'Dashboard',
      hint: 'Progress snapshot',
      route: '/app/dashboard',
      icon: appIcons.LayoutDashboard
    },
    {
      label: 'Chat',
      hint: 'Textbook grounded AI',
      route: '/app/chat',
      icon: appIcons.MessagesSquare
    },
    {
      label: 'Study Tools',
      hint: 'Summaries and notes',
      route: '/app/study-tools',
      icon: appIcons.WandSparkles
    },
    {
      label: 'Flashcards',
      hint: 'Active recall',
      route: '/app/flashcards',
      icon: appIcons.BookOpen
    },
    {
      label: 'Quizzes',
      hint: 'Test mode',
      route: '/app/quizzes',
      icon: appIcons.ClipboardList
    }
  ];

  readonly userNavItems: NavItem[] = [
    {
      label: 'Analytics',
      hint: 'Retention trends',
      route: '/app/analytics',
      icon: appIcons.ChartColumnBig
    },
    {
      label: 'Profile',
      hint: 'Identity and goals',
      route: '/app/profile',
      icon: appIcons.UserRound
    },
    {
      label: 'Settings',
      hint: 'Theme and access',
      route: '/app/settings',
      icon: appIcons.Settings2
    }
  ];
}
