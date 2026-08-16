import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DashboardShell } from '@/components/DashboardShell';

const push = jest.fn();

jest.mock('next/navigation', () => ({
  useRouter: () => ({ push }),
  usePathname: () => '/',
}));

describe('DashboardShell navigation', () => {
  beforeEach(() => {
    push.mockReset();
  });

  it('renders the contracts tab', () => {
    render(
      <DashboardShell title="Test Dashboard">
        <div>content</div>
      </DashboardShell>
    );

    expect(screen.getByRole('link', { name: /contracts/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /campaign/i })).toBeInTheDocument();
  });

  it('routes to the contracts tab on C shortcut', async () => {
    const user = userEvent.setup();

    render(
      <DashboardShell title="Test Dashboard">
        <div>content</div>
      </DashboardShell>
    );

    await user.keyboard('C');

    expect(push).toHaveBeenCalledWith('/contracts');
  });

  it('routes to the campaign tab on A shortcut', async () => {
    const user = userEvent.setup();

    render(
      <DashboardShell title="Test Dashboard">
        <div>content</div>
      </DashboardShell>
    );

    await user.keyboard('A');

    expect(push).toHaveBeenCalledWith('/campaign');
  });
});
