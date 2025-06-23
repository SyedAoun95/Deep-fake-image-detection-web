
import { motion } from "framer-motion";
import { ShieldAlert, Eye, Brain, Zap } from "lucide-react";

const AboutSection = () => {
  const features = [
    {
      icon: ShieldAlert,
      title: "What are Deepfakes?",
      description: "Deepfakes are AI-generated or manipulated media where a person's likeness is replaced with someone else's. Created using deep learning technology, they can produce realistic but entirely fabricated images, videos, or audio.",
    },
    {
      icon: Eye,
      title: "Why Detection Matters",
      description: "As deepfake technology advances, distinguishing real from fake becomes increasingly difficult. This poses serious threats to information integrity, personal reputation, and even political stability and national security.",
    },
    {
      icon: Brain,
      title: "Our Technology",
      description: "We use advanced neural networks and computer vision techniques to analyze images at the pixel level, identifying inconsistencies and artifacts that are typical signatures of deepfake manipulation.",
    },
    {
      icon: Zap,
      title: "Stay Protected",
      description: "Being able to verify the authenticity of visual media is crucial in today's digital landscape. Our tool helps you make informed decisions about the content you consume and share online.",
    },
  ];

  return (
    <section id="about" className="py-20">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold font-orbitron mb-4 text-white">
            About <span className="text-neon-purple neon-glow">Deepfakes</span>
          </h2>
          <p className="text-gray-300 max-w-2xl mx-auto">
            Understanding deepfakes is the first step toward combating their potential for harm.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="glass rounded-xl p-6 relative overflow-hidden"
            >
              <div className="mb-4">
                <div className="w-12 h-12 rounded-lg flex items-center justify-center bg-gradient-to-r from-neon-purple to-neon-blue">
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
              </div>
              
              <h3 className="text-xl font-bold mb-3 text-white font-orbitron">
                {feature.title}
              </h3>
              
              <p className="text-gray-400">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-16 text-center"
        >
          <div className="glass rounded-xl p-8 max-w-3xl mx-auto">
            <h3 className="text-xl font-bold mb-4 text-white font-orbitron">
              The Future of Media Authentication
            </h3>
            <p className="text-gray-300">
              As AI-generated content becomes more advanced and prevalent, tools that can verify authenticity will become essential. Our mission is to help maintain trust in digital media through accessible, powerful verification technology.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default AboutSection;
