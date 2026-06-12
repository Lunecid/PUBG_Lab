import { ReactNode } from "react";

interface Props {
  title: ReactNode;
  kicker?: ReactNode;
  children: ReactNode;
  id?: string;
}

export default function Section({ title, kicker, children, id }: Props) {
  return (
    <section id={id} className="scroll-mt-20">
      {kicker ? (
        <div className="jet mb-1 text-[10px] uppercase tracking-wider text-accent">
          {kicker}
        </div>
      ) : null}
      <h2 className="news mb-4 text-2xl font-semibold leading-tight">
        {title}
      </h2>
      <div className="space-y-4">{children}</div>
    </section>
  );
}
