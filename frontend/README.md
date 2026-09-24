# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

## AWS WebSocket config

Copy the example env file and set your API WebSocket URL:

```bash
cp .env.example .env.local
```

Edit `.env.local` and set:

```
VITE_WS_URL=ws://<your-ec2-ip>/ws
VITE_API_URL=http://<your-ec2-ip>
VITE_DEVICE_ID=rpi-1
```

## Build and run

```bash
npm install
npm run dev
```

For production:

```bash
npm run build
```

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
