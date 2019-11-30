# Machine Learning Experiments Demo

## TODO

- Add route titles using `Helmet`
- Cleanup this README

## How to use this repo

### Linting

To check source code by ESLint run:

```bash
yarn lint
```

To fix the code according to ESLint config run:

```bash
yarn lint --fix
```

### Flow type checking

```bash
yarn flow
```

This project is using [Flow](https://flow.org/) type checking. If you're using VSCode you might want to disable a javascript validation by adding this line to you VCode config file:

```json
"javascript.validate.enable": false,
```

This is because at the moment of writing this file the VSCode [didn't support](https://github.com/Microsoft/vscode-react-native/issues/631) Flow types correctly.

### Adding new flow type definitions

First you may want to search for a definitions:

```bash
yarn flow-typed search your-package-name
```

If definitions are not created you may want to mock the module:

```bash
yarn flow-typed create-stub your-package-name
```

If definitions exist you may install them:

```bash
yarn flow-typed install your-package-name
```

## Available Scripts

In the project directory, you can run:

### `yarn start`

Runs the app in the development mode.

Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.

You will also see any lint errors in the console.

### `yarn test`

Launches the test runner in the interactive watch mode.

See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `yarn build`

Builds the app for production to the `build` folder.

It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.

Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `yarn deploy`

Builds the demo project and pushes the build to `gh-pages` branch.

### `yarn eject`

**Note: this is a one-way operation. Once you `eject`, you can’t go back!**

If you aren’t satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (Webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you’re on your own.

You don’t have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn’t feel obligated to use this feature. However we understand that this tool wouldn’t be useful if you couldn’t customize it when you are ready for it.
