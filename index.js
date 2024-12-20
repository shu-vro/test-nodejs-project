import express from "express";
import dotenv from "dotenv";
import {
    GoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
} from "@google/generative-ai";
import morgan from "morgan";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import { query, validationResult } from "express-validator";

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;
const apiKey = process.env.GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(apiKey);

const model = genAI.getGenerativeModel({
    model: "tunedModels/generate-num-2948",
    safetySettings: [
        {
            category: HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
    ],
    // systemInstruction: "Increment the number by 1, regardless of the language",
});

const generationConfig = {
    temperature: 1,
    topP: 0.95,
    topK: 40,
    maxOutputTokens: 8192,
    responseMimeType: "text/plain",
};

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(
    morgan('[:date[clf]] ":method :url" :remote-addr :status :response-time ms')
);
app.use(helmet());
app.use("/static", express.static("public"));

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100,
});
app.use(limiter);

app.get("/", (req, res) => {
    res.send("Go to /increment?num=1 to increment the number by 1");
});

// Define a route handler for the increment endpoint
app.get(
    "/increment",
    [
        query("num")
            .notEmpty()
            .withMessage("The 'num' parameter is required and cannot be empty"),
    ],
    async (req, res) => {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res
                .status(400)
                .json({ errors: errors.array(), status: "error" });
        }

        try {
            const chatSession = model.startChat({
                generationConfig,
                history: [],
            });

            const result = await chatSession.sendMessage(req.query.num);
            res.json({
                query: req.query.num,
                response: result.response.text(),
                status: "success",
            });
        } catch (error) {
            console.error(error);
            res.status(500).send("Internal Server Error");
        }
    }
);

// Start the Express server
app.listen(port, () => {
    console.log(`Server started at http://localhost:${port}`);
});
