import { Suspense, lazy } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";

const Home = lazy(() => import("./routes/Home"));
const Network = lazy(() => import("./routes/Network"));
const Strategy = lazy(() => import("./routes/Strategy"));
const Phases = lazy(() => import("./routes/Phases"));
const Hypotheses = lazy(() => import("./routes/Hypotheses"));
const Explorer = lazy(() => import("./routes/Explorer"));
const MatchDetail = lazy(() => import("./routes/MatchDetail"));
const Methods = lazy(() => import("./routes/Methods"));
const Glossary = lazy(() => import("./routes/Glossary"));

function PageFallback() {
  return (
    <div className="flex h-64 items-center justify-center text-muted">
      <div className="animate-pulse text-sm">불러오는 중…</div>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/network" element={<Network />} />
          <Route path="/strategy" element={<Strategy />} />
          <Route path="/phases" element={<Phases />} />
          <Route path="/hypotheses" element={<Hypotheses />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/explorer/:matchId" element={<MatchDetail />} />
          <Route path="/methods" element={<Methods />} />
          <Route path="/glossary" element={<Glossary />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
