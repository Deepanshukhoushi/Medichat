import { Injectable } from '@angular/core';

import { appIcons } from '../../shared/icons/lucide-icons';
import { NavItem } from '../../shared/models/navigation.model';

@Injectable({ providedIn: 'root' })
export class NavigationService {
  readonly navItems: NavItem[] = [
    {
      label: 'Continue Learning',
      hint: 'Your study dashboard',
      route: '/app/dashboard',
      icon: appIcons.LayoutDashboard
    },
    {
      label: 'Collections',
      hint: 'Saved topics and bookmarks',
      route: '/app/collections',
      icon: appIcons.LibraryBig
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
    },
    {
      label: 'Notes',
      hint: 'Your generated summaries',
      route: '/app/notes',
      icon: appIcons.FileText
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
