import { mebibytesToBytes } from '@/lib/utils';

describe('mebibytesToBytes', () => {
  it('converts nvidia-smi MiB values with the binary multiplier', () => {
    expect(mebibytesToBytes(1024)).toBe(1024 ** 3);
    expect(mebibytesToBytes(512)).toBe(512 * 1024 ** 2);
  });
});
