import { lazy, type ReactNode, Suspense } from 'react';
import { NavLink, Route, Routes, useLocation } from 'react-router-dom';
import { IconChart, IconHistory, IconHome, IconLab, IconSettings, IconTarget } from './components/icons';
import Home from './pages/Home';

const Workouts = lazy(() => import('./pages/Workouts'));
const Calibrate = lazy(() => import('./pages/Calibrate'));
const Train = lazy(() => import('./pages/Train'));
const Results = lazy(() => import('./pages/Results'));
const History = lazy(() => import('./pages/History'));
const Analytics = lazy(() => import('./pages/Analytics'));
const SettingsPage = lazy(() => import('./pages/Settings'));
const Lab = lazy(() => import('./pages/Lab'));

const NAV = [
  { to: '/', label: 'Home', icon: IconHome, end: true },
  { to: '/workouts', label: 'Train', icon: IconTarget },
  { to: '/history', label: 'History', icon: IconHistory },
  { to: '/analytics', label: 'Analytics', icon: IconChart },
  { to: '/settings', label: 'Settings', icon: IconSettings },
];

function Brand() {
  return (
    <div className="flex items-center gap-2">
      <svg width="30" height="30" viewBox="0 0 64 64" aria-hidden>
        <circle cx="32" cy="36" r="17" fill="none" stroke="#ff6b00" strokeWidth="5" />
        <path d="M15 36h34M32 19v34" stroke="#ff6b00" strokeWidth="3.5" />
        <ellipse cx="32" cy="12" rx="13" ry="4" fill="none" stroke="#faf8f3" strokeWidth="4" />
      </svg>
      <span className="font-display text-2xl font-800 uppercase tracking-tight">
        Hoop<span className="text-ember-500">Vision</span>
      </span>
    </div>
  );
}

function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-full flex-col md:flex-row">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-coal-800 bg-coal-950 px-4 py-6 md:flex">
        <Brand />
        <nav className="mt-8 flex flex-col gap-1">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-semibold transition ${isActive ? 'bg-coal-800 text-ember-400' : 'text-chalk-300 hover:text-chalk-50'}`
              }
            >
              <n.icon /> {n.label}
            </NavLink>
          ))}
          <NavLink
            to="/lab"
            className={({ isActive }) =>
              `mt-4 flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-semibold transition ${isActive ? 'bg-coal-800 text-ember-400' : 'text-chalk-400 hover:text-chalk-50'}`
            }
          >
            <IconLab /> Vision Lab
          </NavLink>
        </nav>
        <p className="mt-auto text-[11px] leading-relaxed text-chalk-400">All video is processed on this device. Nothing is uploaded.</p>
      </aside>
      <header className="flex items-center justify-between border-b border-coal-800 px-4 py-3 md:hidden">
        <Brand />
        <NavLink to="/lab" className="btn-ghost px-2 py-2" aria-label="Vision Lab">
          <IconLab />
        </NavLink>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 pb-28 pt-6 md:px-8 md:pb-10 md:pt-10">{children}</main>
      <nav className="fixed inset-x-0 bottom-0 z-40 grid grid-cols-5 border-t border-coal-800 bg-coal-950/95 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden">
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.end}
            className={({ isActive }) => `flex flex-col items-center gap-1 py-2.5 text-[10px] font-semibold uppercase tracking-wide ${isActive ? 'text-ember-400' : 'text-chalk-400'}`}
          >
            <n.icon size={22} />
            {n.label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

const Loading = () => <div className="flex h-64 items-center justify-center text-sm text-chalk-400">Loading…</div>;

export default function App() {
  const loc = useLocation();
  // full-bleed screens without navigation chrome
  const immersive = loc.pathname.startsWith('/train') || loc.pathname.startsWith('/calibrate');
  const routes = (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/workouts" element={<Workouts />} />
        <Route path="/calibrate" element={<Calibrate />} />
        <Route path="/train" element={<Train />} />
        <Route path="/results/:id" element={<Results />} />
        <Route path="/history" element={<History />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/lab" element={<Lab />} />
        <Route path="*" element={<Home />} />
      </Routes>
    </Suspense>
  );
  return immersive ? routes : <Shell>{routes}</Shell>;
}
