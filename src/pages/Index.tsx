
import { useEffect } from "react";
import { motion } from "framer-motion";
import NavBar from "@/components/NavBar";
import HeroSection from "@/components/HeroSection";
import UploadSection from "@/components/UploadSection";
import HowToUseSection from "@/components/HowToUseSection";
import AboutSection from "@/components/AboutSection";
import Footer from "@/components/Footer";

const Index = () => {
  // Set dark background and remove any overflow constraints
  useEffect(() => {
    document.body.style.overflow = "auto";
    
    return () => {
      document.body.style.overflow = "";
    };
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <NavBar />
      
      {/* Main Content */}
      <main className="relative">
        <HeroSection />
        <UploadSection />
        <HowToUseSection />
        <AboutSection />
      </main>
      
      <Footer />
    </div>
  );
};

export default Index;
