/** @jest-environment node */

import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, sep } from 'node:path';
import type { Rule } from 'eslint';
import { getRootDirs } from '@next/eslint-plugin-next/dist/utils/get-root-dirs';

let workspace: string;

function createDirectory(relativePath: string) {
  const path = join(workspace, relativePath);
  mkdirSync(path, { recursive: true });
  return path;
}

function resolveRootDirs(rootDir: string | string[]) {
  const context = {
    cwd: workspace,
    settings: { next: { rootDir } },
  } as unknown as Rule.RuleContext;

  return getRootDirs(context).sort();
}

beforeAll(() => {
  workspace = mkdtempSync(join(tmpdir(), 'next-eslint-root-dirs-'));
  createDirectory('apps/alpha');
  createDirectory('apps/beta');
  createDirectory('packages/docs');
  createDirectory('ranges/app-1');
  createDirectory('ranges/app-2');
  createDirectory('ranges/app-3');
  const linkedApp = createDirectory('shared/linked-app');
  symlinkSync(linkedApp, join(workspace, 'apps', 'linked-app'), 'dir');
  writeFileSync(join(workspace, 'apps', 'README.md'), 'not a directory\n');
});

afterAll(() => {
  rmSync(workspace, { recursive: true, force: true });
});

describe('Next ESLint root directory glob backport', () => {
  it('resolves literal, wildcard, and brace patterns', () => {
    expect(resolveRootDirs(join(workspace, 'apps', 'alpha'))).toEqual([
      join(workspace, 'apps', 'alpha'),
    ]);
    expect(resolveRootDirs(join(workspace, 'apps', '*'))).toEqual(
      [
        join(workspace, 'apps', 'alpha'),
        join(workspace, 'apps', 'beta'),
        join(workspace, 'apps', 'linked-app'),
      ].sort()
    );
    expect(resolveRootDirs(join(workspace, 'apps', '{alpha,beta}'))).toEqual(
      [join(workspace, 'apps', 'alpha'), join(workspace, 'apps', 'beta')].sort()
    );
  });

  it('supports Windows separators and arrays of roots', () => {
    const windowsPattern = join(workspace, 'apps', 'alpha').split(sep).join('\\');
    expect(resolveRootDirs(windowsPattern)).toEqual([join(workspace, 'apps', 'alpha')]);
    expect(
      resolveRootDirs([
        join(workspace, 'apps', 'beta'),
        join(workspace, 'packages', 'docs'),
      ])
    ).toEqual(
      [join(workspace, 'apps', 'beta'), join(workspace, 'packages', 'docs')].sort()
    );
  });

  it('follows matching directory links and excludes files and missing roots', () => {
    expect(resolveRootDirs(join(workspace, 'apps', 'linked-*'))).toEqual([
      join(workspace, 'apps', 'linked-app'),
    ]);
    expect(resolveRootDirs(join(workspace, 'apps', 'README.md'))).toEqual([]);
    expect(resolveRootDirs(join(workspace, 'apps', 'missing'))).toEqual([]);
  });

  it('preserves stepped range patterns', () => {
    expect(resolveRootDirs(join(workspace, 'ranges', 'app-{1..3..2}'))).toEqual(
      [join(workspace, 'ranges', 'app-1'), join(workspace, 'ranges', 'app-3')].sort()
    );
  });
});
