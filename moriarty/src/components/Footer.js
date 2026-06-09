import Link from "next/link";
import React from "react";
import Layout from "./Layout";

const Footer = () => {
  return (
    <footer className="w-full bg-dark text-light text-sm font-medium sm:text-xs">
      <Layout className="py-3 flex items-center justify-center">
        <span>
          {new Date().getFullYear()} &copy; All Rights Reserved. Built by&nbsp;
          <Link
            href="https://github.com/fullscreen-triangle"
            target="_blank"
            className="underline underline-offset-2"
          >
            Fullscreen Triangle
          </Link>
        </span>
      </Layout>
    </footer>
  );
};

export default Footer;