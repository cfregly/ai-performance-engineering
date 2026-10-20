import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';

// Backport source: https://github.com/vercel/next.js/pull/100264
// Upstream commit: 13442125309b8976982d69b5cde435ab1fe6c0cc
// Remove this script and override after a stable @next/eslint-plugin-next release removes fast-glob.
const pluginVersion = '16.4.2';
const globVersion = '13.0.6';
const originalSourceHash = '886677432990a735e5ebdfb345ff1cbd40e264f9947a8432eeeb254b3a926bde';
const patchedSourceHash = 'e082104c7a4b6c8d2e1d1b3fe6a3cf57b277c2bc4d72a19876a05126f033c2cf';
const require = createRequire(import.meta.url);

function parsePackage(source) {
  return JSON.parse(source);
}

function sha256(source) {
  return createHash('sha256').update(source).digest('hex');
}

function isMissingModule(error, request) {
  return (
    error?.code === 'MODULE_NOT_FOUND' &&
    typeof error.message === 'string' &&
    error.message.includes(request)
  );
}

function configListIncludes(name, value) {
  return (value ?? '')
    .split(/[\s,]+/)
    .filter(Boolean)
    .includes(name);
}

function isProductionOnlyInstall() {
  const installCommand = process.env.npm_command === 'ci' || process.env.npm_command === 'install';
  const omitsDev =
    configListIncludes('dev', process.env.npm_config_omit) || process.env.NODE_ENV === 'production';
  const includesDev = configListIncludes('dev', process.env.npm_config_include);
  return installCommand && omitsDev && !includesDev;
}

let pluginPackagePath;
try {
  pluginPackagePath = require.resolve('@next/eslint-plugin-next/package.json');
} catch (error) {
  if (!isMissingModule(error, '@next/eslint-plugin-next/package.json')) {
    throw error;
  }

  let eslintConfigIsMissing = false;
  try {
    require.resolve('eslint-config-next');
  } catch (eslintConfigError) {
    if (!isMissingModule(eslintConfigError, 'eslint-config-next')) {
      throw eslintConfigError;
    }
    eslintConfigIsMissing = true;
  }

  if (!eslintConfigIsMissing || !isProductionOnlyInstall()) {
    throw error;
  }

  console.log(
    'Skipping the Next ESLint glob backport because npm omitted development dependencies for this production install.'
  );
  process.exit(0);
}

const pluginPackage = parsePackage(await readFile(pluginPackagePath, 'utf8'));

if (pluginPackage.name !== '@next/eslint-plugin-next' || pluginPackage.version !== pluginVersion) {
  throw new Error(
    `Expected @next/eslint-plugin-next ${pluginVersion}, received ${pluginPackage.name} ${pluginPackage.version}`
  );
}
if (pluginPackage.dependencies?.['fast-glob'] !== '3.3.1') {
  throw new Error(
    `Expected the audited fast-glob dependency 3.3.1, received ${pluginPackage.dependencies?.['fast-glob']}`
  );
}

const pluginRequire = createRequire(pluginPackagePath);
const globPackagePath = pluginRequire.resolve('fast-glob/package.json');
const globPackage = parsePackage(await readFile(globPackagePath, 'utf8'));

if (globPackage.name !== 'glob' || globPackage.version !== globVersion) {
  throw new Error(
    `Expected the fast-glob alias to resolve to glob ${globVersion}, received ${globPackage.name} ${globPackage.version}`
  );
}

const modulePath = require.resolve('@next/eslint-plugin-next/dist/utils/get-root-dirs');
const source = await readFile(modulePath, 'utf8');
const sourceHash = sha256(source);
const before = [
  "    return (0, _fastglob.globSync)(rootDir.replace(/\\\\/g, '/'), {",
  '        onlyDirectories: true',
  '    });',
].join('\n');
const after = [
  '    return (0, _fastglob.globSync)(`${rootDir}/`, {',
  '        follow: true,',
  '        nocase: false,',
  '        windowsPathsNoEscape: true',
  '    });',
].join('\n');

if (sourceHash === patchedSourceHash) {
  if (!source.includes(after) || source.includes(before)) {
    throw new Error('The patched Next ESLint helper has unexpected content');
  }
  process.exit(0);
}
if (sourceHash !== originalSourceHash) {
  throw new Error(`Next ESLint helper source hash changed to ${sourceHash}`);
}
if (!source.includes(before) || source.includes(after)) {
  throw new Error('Next ESLint helper no longer matches the audited upstream patch target');
}

const patchedSource = source.replace(before, after);
if (sha256(patchedSource) !== patchedSourceHash) {
  throw new Error('The Next ESLint helper patch produced unexpected content');
}

await writeFile(modulePath, patchedSource);
console.log(`Applied Next ESLint glob backport for @next/eslint-plugin-next ${pluginVersion}`);
