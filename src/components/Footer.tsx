
import { motion } from "framer-motion";
import { Github, Twitter, Linkedin } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-10 border-t border-gray-800">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col space-y-4"
          >
            <h3 className="font-orbitron text-xl font-bold text-white">
              <span className="text-neon-purple">Deepfake</span> Detector
            </h3>
            <p className="text-gray-400 text-sm">
              Defending digital truth through advanced AI technology. Our tool helps identify manipulated images with precision and confidence.
            </p>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex flex-col space-y-4"
          >
            <h3 className="font-orbitron text-lg font-semibold text-white">Links</h3>
            <nav className="flex flex-col space-y-2">
              <a href="#hero" className="text-gray-400 hover:text-neon-purple transition-colors">Home</a>
              <a href="#upload" className="text-gray-400 hover:text-neon-purple transition-colors">Upload</a>
              <a href="#howto" className="text-gray-400 hover:text-neon-purple transition-colors">How to Use</a>
              <a href="#about" className="text-gray-400 hover:text-neon-purple transition-colors">About</a>
            </nav>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="flex flex-col space-y-4"
          >
            <h3 className="font-orbitron text-lg font-semibold text-white">Connect</h3>
            <div className="flex space-x-4">
              <a href="https://github.com/sumiitttt11" target="_blank" className="text-gray-400 hover:text-neon-purple transition-colors">
                <Github size={20} />
              </a>
              <a href="https://x.com/sumiitttt11" target="_blank" className="text-gray-400 hover:text-neon-purple transition-colors">
                <Twitter size={20} />
              </a>
              <a href="https://www.linkedin.com/in/sumiitttt11/" target="_blank" className="text-gray-400 hover:text-neon-purple transition-colors">
                <Linkedin size={20} />
              </a>
            </div>
          </motion.div>
        </div>
        
        <div className="mt-8 pt-6 border-t border-gray-800 flex flex-col md:flex-row justify-between items-center text-gray-500 text-sm">
          <p>© 2025 Deepfake Detector. All rights reserved.</p>
          <div className="flex space-x-6 mt-4 md:mt-0">
            <a href="#" className="hover:text-gray-300">Privacy Policy</a>
            <a href="#" className="hover:text-gray-300">Terms of Service</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
