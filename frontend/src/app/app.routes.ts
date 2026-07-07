import { Routes } from '@angular/router';
import { authGuard } from './core/guards/auth.guard';
import { alreadyAuthGuard } from './core/guards/already-auth.guard';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./layouts/marketing-layout/marketing-layout.component').then((m) => m.MarketingLayoutComponent),
    children: [
      {
        path: '',
        pathMatch: 'full',
        title: 'MediChat - Medical AI Assistant',
        loadComponent: () => import('./features/landing/landing-page/landing-page.component').then((m) => m.LandingPageComponent)
      }
    ]
  },
  {
    path: 'auth',
    canActivate: [alreadyAuthGuard],
    loadComponent: () => import('./layouts/auth-layout/auth-layout.component').then((m) => m.AuthLayoutComponent),
    children: [
      {
        path: 'login',
        title: 'Log In | MediChat',
        loadComponent: () => import('./features/auth/login-page/login-page.component').then((m) => m.LoginPageComponent)
      },
      {
        path: 'signup',
        title: 'Sign Up | MediChat',
        loadComponent: () => import('./features/auth/signup-page/signup-page.component').then((m) => m.SignupPageComponent)
      },
      {
        path: 'forgot-password',
        title: 'Forgot Password | MediChat',
        loadComponent: () => import('./features/auth/forgot-password-page/forgot-password-page.component').then((m) => m.ForgotPasswordPageComponent)
      },
      {
        path: 'reset-password',
        title: 'Reset Password | MediChat',
        loadComponent: () => import('./features/auth/reset-password-page/reset-password-page.component').then((m) => m.ResetPasswordPageComponent)
      },
      {
        path: '',
        pathMatch: 'full',
        redirectTo: 'login'
      }

    ]
  },
  {
    path: 'app',
    pathMatch: 'full',
    redirectTo: 'app/chat'
  },
  {
    path: 'app',
    canActivate: [authGuard],
    loadComponent: () => import('./layouts/app-shell-layout/app-shell-layout.component').then((m) => m.AppShellLayoutComponent),
    children: [
      {
        path: 'dashboard',
        title: 'Dashboard | MediChat',
        loadComponent: () => import('./features/dashboard/dashboard-page/dashboard-page.component').then((m) => m.DashboardPageComponent)
      },
      {
        path: 'chat',
        title: 'Chat | MediChat',
        data: { allowGuest: true },
        loadComponent: () => import('./features/chat/chat-page/chat-page.component').then((m) => m.ChatPageComponent)
      },
      {
        path: 'study-tools',
        title: 'Study Tools | MediChat',
        data: { allowGuest: true },
        loadComponent: () => import('./features/study-tools/study-tools-page/study-tools-page.component').then((m) => m.StudyToolsPageComponent)
      },
      {
        path: 'flashcards',
        title: 'Flashcards | MediChat',
        loadComponent: () => import('./features/flashcards/flashcards-page/flashcards-page.component').then((m) => m.FlashcardsPageComponent)
      },
      {
        path: 'quizzes',
        title: 'Quizzes | MediChat',
        loadComponent: () => import('./features/quizzes/quizzes-page/quizzes-page.component').then((m) => m.QuizzesPageComponent)
      },
      {
        path: 'profile',
        title: 'Profile | MediChat',
        loadComponent: () => import('./features/profile/profile-page/profile-page.component').then((m) => m.ProfilePageComponent)
      },
      {
        path: 'settings',
        title: 'Settings | MediChat',
        loadComponent: () => import('./features/settings/settings-page/settings-page.component').then((m) => m.SettingsPageComponent)
      },
      {
        path: 'analytics',
        title: 'Analytics | MediChat',
        loadComponent: () => import('./features/analytics/analytics-page/analytics-page.component').then((m) => m.AnalyticsPageComponent)
      }
    ]
  },
  {
    path: '**',
    title: 'Page not found | MediChat',
    loadComponent: () => import('./features/not-found/not-found-page/not-found-page.component').then((m) => m.NotFoundPageComponent)
  }
];
