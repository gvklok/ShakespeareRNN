# Build stage
FROM node:16-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies (including serve)
RUN npm ci

# Copy application code
COPY . .

# Accept build argument and set as env var for React build
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=$REACT_APP_API_URL

# Build the app
RUN npm run build

# Install serve globally
RUN npm install -g serve

# Expose port (Railway uses $PORT dynamically)
EXPOSE 3000

# Start with serve on dynamic port
CMD serve -s build -l $PORT
