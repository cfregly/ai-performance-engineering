import { act, renderHook } from '@testing-library/react';
import { useGpuStream } from './useGpuStream';

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly url: string;
  onerror: ((event: Event) => void) | null = null;
  closed = false;
  private listeners = new Map<string, (event: MessageEvent) => void>();

  constructor(url: string | URL) {
    this.url = String(url);
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    this.listeners.set(type, listener as (event: MessageEvent) => void);
  }

  emit(type: string, data: unknown) {
    this.listeners.get(type)?.({ data: JSON.stringify(data) } as MessageEvent);
  }

  close() {
    this.closed = true;
  }
}

describe('useGpuStream', () => {
  beforeEach(() => {
    FakeEventSource.instances = [];
    global.EventSource = FakeEventSource as unknown as typeof EventSource;
  });

  it('marks the last sample stale when the stream errors', () => {
    const { result, unmount } = renderHook(() => useGpuStream({ intervalMs: 5000 }));
    const source = FakeEventSource.instances[0];

    act(() => {
      source.emit('gpu', {
        timestamp: '2026-09-05T00:00:00Z',
        gpu: {
          name: 'Test GPU',
          memory_total: 1024,
          memory_used: 512,
          utilization: 50,
          temperature: 40,
        },
      });
    });
    expect(result.current.status).toBe('connected');
    expect(result.current.gpu?.live).toBe(true);

    act(() => {
      source.onerror?.(new Event('error'));
    });
    expect(result.current.status).toBe('error');
    expect(result.current.gpu?.live).toBe(false);

    unmount();
    expect(source.closed).toBe(true);
  });
});
