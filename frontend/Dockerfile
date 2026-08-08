# Base builder stage
FROM node:20-alpine AS builder

WORKDIR /workspace

COPY package.json /workspace/
RUN npm install

COPY . /workspace/
RUN npm run build

# Runner stage
FROM node:20-alpine AS runner

WORKDIR /workspace

COPY --from=builder /workspace/package.json /workspace/
COPY --from=builder /workspace/node_modules /workspace/node_modules
COPY --from=builder /workspace/.next /workspace/.next
COPY --from=builder /workspace/public /workspace/public
COPY --from=builder /workspace/src /workspace/src

EXPOSE 3000

CMD ["npm", "run", "start"]
