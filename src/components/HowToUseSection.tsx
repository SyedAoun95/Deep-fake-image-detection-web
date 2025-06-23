
import { motion } from "framer-motion";
import { Upload, Search, AlertCircle, CheckCircle } from "lucide-react";

const HowToUseSection = () => {
  const steps = [
    {
      id: 1,
      title: "Upload",
      description: "Upload any suspicious image you want to analyze. We support JPG, PNG, and other common formats.",
      icon: Upload,
      color: "from-neon-purple to-neon-pink",
    },
    {
      id: 2,
      title: "Analyze",
      description: "Our AI algorithm examines the image for telltale signs of manipulation that are invisible to the human eye.",
      icon: Search,
      color: "from-neon-blue to-neon-purple",
    },
    {
      id: 3,
      title: "Review",
      description: "The system identifies potential anomalies and highlights areas that may have been artificially generated or altered.",
      icon: AlertCircle,
      color: "from-neon-pink to-neon-purple",
    },
    {
      id: 4,
      title: "Verify",
      description: "Get detailed results about the authenticity of the image with a confidence score and explanation.",
      icon: CheckCircle,
      color: "from-neon-purple to-neon-blue",
    },
  ];

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5 } },
  };

  return (
    <section id="howto" className="py-20 bg-black/30">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold font-orbitron mb-4 text-white">
            How To <span className="text-neon-purple neon-glow">Use</span>
          </h2>
          <p className="text-gray-300 max-w-2xl mx-auto">
            Follow these simple steps to identify potential deepfakes and protect yourself from misinformation.
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          className="grid md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {steps.map((step) => (
            <motion.div
              key={step.id}
              variants={item}
              className="glass rounded-xl p-6 relative overflow-hidden group hover:scale-105 transition-transform duration-300"
            >
              <div className={`absolute inset-0 bg-gradient-to-br ${step.color} opacity-0 group-hover:opacity-10 transition-opacity duration-300`} />
              
              <div className="mb-4 relative">
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center bg-gradient-to-r ${step.color}`}>
                  <step.icon className="w-6 h-6 text-white" />
                </div>
                <div className="absolute -top-2 -right-1 w-8 h-8 rounded-full bg-black/50 flex items-center justify-center border border-white/10 text-white text-sm font-medium">
                  {step.id}
                </div>
              </div>
              
              <h3 className="text-xl font-bold mb-2 text-white font-orbitron">
                {step.title}
              </h3>
              
              <p className="text-gray-400">
                {step.description}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default HowToUseSection;
