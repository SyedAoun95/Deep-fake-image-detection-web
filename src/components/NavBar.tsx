
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const NavBar = () => {
  const [activeSection, setActiveSection] = useState("hero");
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const sections = ["hero", "upload", "howto", "about"];
      const scrollPosition = window.scrollY;

      // Check if scrolled past a threshold to change navbar style
      if (scrollPosition > 50) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }

      // Determine which section is in view
      for (const section of sections) {
        const element = document.getElementById(section);
        if (element) {
          const top = element.offsetTop - 100;
          const bottom = top + element.offsetHeight;
          
          if (scrollPosition >= top && scrollPosition < bottom) {
            setActiveSection(section);
            break;
          }
        }
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (sectionId: string) => {
    const section = document.getElementById(sectionId);
    if (section) {
      window.scrollTo({
        top: section.offsetTop - 80,
        behavior: "smooth",
      });
    }
  };

  const navItems = [
    { id: "hero", label: "Home" },
    { id: "upload", label: "Upload" },
    { id: "howto", label: "How to Use" },
    { id: "about", label: "About" },
  ];

  return (
    <motion.nav 
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300 py-4",
        scrolled ? "glass" : "bg-transparent"
      )}
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: "spring", stiffness: 100, duration: 0.4 }}
    >
      <div className="container mx-auto px-4 flex items-center justify-between">
        <motion.div 
          className="flex items-center"
          whileHover={{ scale: 1.05 }}
        >
          <a
            href="https://deepfake-frontend-tan.vercel.app/"
            rel="noopener noreferrer"
            className="text-xl md:text-2xl font-bold font-orbitron text-white neon-glow"
          >
            <span className="text-neon-purple">Deepfake</span> Detector
          </a>

        </motion.div>
        
        <div className="hidden md:flex space-x-6">
          {navItems.map((item) => (
            <motion.button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              className={cn(
                "text-sm font-medium transition-colors relative",
                activeSection === item.id 
                  ? "text-white" 
                  : "text-slate-400 hover:text-white"
              )}
              whileHover={{ scale: 1.05 }}
            >
              {item.label}
              {activeSection === item.id && (
                <motion.div
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-neon-purple"
                  layoutId="activeSection"
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
            </motion.button>
          ))}
        </div>
        
        <div className="md:hidden">
          <button className="text-white">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </motion.nav>
  );
};

export default NavBar;
