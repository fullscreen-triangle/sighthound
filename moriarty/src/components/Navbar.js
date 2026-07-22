import Link from "next/link";
import React, { useState } from "react";
import Logo from "./Logo";
import { useRouter } from "next/router";
import { motion } from "framer-motion";

const CustomLink = ({ href, title, className = "" }) => {
  const router = useRouter();

  return (
    <Link href={href} className={`${className} rounded relative group text-light`}>
      {title}
      <span
        className={`
              inline-block h-[1px] bg-light absolute left-0 -bottom-0.5
              group-hover:w-full transition-[width] ease duration-300
              ${router.asPath === href ? "w-full" : " w-0"}
              `}
      >
        &nbsp;
      </span>
    </Link>
  );
};

const CustomMobileLink = ({ href, title, className = "", toggle }) => {
  const router = useRouter();

  const handleClick = () => {
    toggle();
    router.push(href);
  };

  return (
    <button
      className={`${className} rounded relative group text-light my-2 text-lg`}
      onClick={handleClick}
    >
      {title}
      <span
        className={`
              inline-block h-[1px] bg-light absolute left-0 -bottom-0.5
              group-hover:w-full transition-[width] ease duration-300
              ${router.asPath === href ? "w-full" : " w-0"}
              `}
      >
        &nbsp;
      </span>
    </button>
  );
};

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const handleClick = () => {
    setIsOpen(!isOpen);
  };

  return (
    <header
      className="w-full flex items-center justify-between px-32 py-8 font-medium z-10 text-light
      lg:px-16 relative md:px-12 sm:px-8"
    >
      {/* mobile hamburger */}
      <button
        type="button"
        className="flex-col items-center justify-center hidden lg:flex"
        aria-controls="mobile-menu"
        aria-expanded={isOpen}
        onClick={handleClick}
      >
        <span className="sr-only">Open main menu</span>
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${isOpen ? "rotate-45 translate-y-1" : "-translate-y-0.5"}`}></span>
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${isOpen ? "opacity-0" : "opacity-100"} my-0.5`}></span>
        <span className={`bg-light block h-0.5 w-6 rounded-sm transition-all duration-300 ease-out ${isOpen ? "-rotate-45 -translate-y-1" : "translate-y-0.5"}`}></span>
      </button>

      {/* desktop nav */}
      <div className="w-full flex justify-between items-center lg:hidden">
        <nav className="flex items-center justify-center">
          <CustomLink className="mx-4" href="/cynegeticus" title="Cynegeticus" />
          <CustomLink className="ml-4" href="/silk" title="Silk" />
        </nav>
      </div>

      {/* mobile overlay menu */}
      {isOpen ? (
        <motion.div
          className="min-w-[70vw] sm:min-w-[90vw] flex justify-between items-center flex-col fixed top-1/2 left-1/2 -translate-x-1/2
          -translate-y-1/2 py-32 bg-dark/95 rounded-lg z-50 backdrop-blur-md border border-light/10"
          initial={{ scale: 0, x: "-50%", y: "-50%", opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
        >
          <nav className="flex items-center justify-center flex-col">
            <CustomMobileLink toggle={handleClick} href="/cynegeticus" title="Cynegeticus" />
            <CustomMobileLink toggle={handleClick} href="/silk" title="Silk" />
          </nav>
        </motion.div>
      ) : null}

      <div className="absolute left-[50%] top-2 translate-x-[-50%]">
        <Logo />
      </div>
    </header>
  );
};

export default Navbar;
