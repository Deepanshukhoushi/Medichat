import { Injectable } from '@angular/core';

import { appIcons } from '../../shared/icons/lucide-icons';
import { NavItem } from '../../shared/models/navigation.model';

@Injectable({ providedIn: 'root' })
export class NavigationService {
  readonly navItems: NavItem[] = [
    {
      label: 'Dashboard',
      hint: 'Your study dashboard',
      route: '/app/dashboard',
      icon: appIcons.LayoutDashboard
    },
    {
      label: 'Flashcards',
      hint: 'Active recall decks',
      route: '/app/flashcards',
      icon: appIcons.BookOpen
    },
    {
      label: 'Quizzes',
      hint: 'Test your knowledge',
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
