
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

const HeroSection = () => {
  const scrollToUpload = () => {
    const uploadSection = document.getElementById("upload");
    if (uploadSection) {
      window.scrollTo({
        top: uploadSection.offsetTop - 80,
        behavior: "smooth",
      });
    }
  };

  return (
    <section id="hero" className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden py-20">
      {/* Background effects */}
      <div className="absolute inset-0 bg-cyber-grid opacity-10 z-0"></div>
      
      <div className="container mx-auto px-4 z-10 text-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
          className="mb-8 relative"
        >
          <motion.div 
            className="w-24 h-24 rounded-full bg-gradient-to-r from-neon-purple/30 to-neon-pink/30 absolute left-1/2 -translate-x-1/2 -top-12 blur-3xl"
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [0.4, 0.7, 0.4]
            }}
            transition={{ 
              duration: 4, 
              repeat: Infinity,
              ease: "easeInOut" 
            }}
          />
          <span className="text-sm uppercase tracking-widest font-medium text-neon-purple/80">
            AI-Powered Technology
          </span>
        </motion.div>

        <motion.h1 
          className="text-4xl md:text-6xl lg:text-7xl font-bold font-orbitron mb-6 leading-tight tracking-tight"
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.2 }}
        >
          <span className="text-white">Detect Deepfakes.</span><br />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-neon-purple to-neon-pink animate-text-shimmer bg-[length:200%_auto] neon-glow">
            Defend Reality.
          </span>
        </motion.h1>

        <motion.p
          className="text-lg md:text-xl text-gray-300 max-w-2xl mx-auto mb-10"
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.4 }}
        >
          Our advanced AI technology helps you identify manipulated images. Stay one step ahead in the age of digital deception.
        </motion.p>

        <motion.div
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.6 }}
        >
          <Button 
            onClick={scrollToUpload}
            className="bg-neon-purple hover:bg-neon-purple/80 text-white px-8 py-6 rounded-lg font-medium text-lg animate-glow transition-all duration-300"
            size="lg"
          >
            Upload Now
          </Button>
        </motion.div>
      </div>

      <motion.div
        className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
        animate={{ y: [0, -10, 0] }}
        transition={{ duration: 1.5, repeat: Infinity }}
      >
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          width="24" 
          height="24" 
          viewBox="0 0 24 24" 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round" 
          className="text-gray-400"
        >
          <path d="M12 5v14"></path>
          <path d="m19 12-7 7-7-7"></path>
        </svg>
      </motion.div>
    </section>
  );
};

export default HeroSection;
