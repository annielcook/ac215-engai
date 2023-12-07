#FROM node:16-alpine as build

#WORKDIR /app
#ENV PATH /app/node_modules/.bin:$PATH
#ENV PUBLIC_URL /

#COPY package.json ./
#COPY yarn.lock ./
#RUN yarn install

#COPY . ./
#RUN yarn run build
#RUN yarn start

# Nginx wrapper to serve static files
FROM nginx:stable
#COPY --from=build /app/build /usr/share/nginx/html
COPY build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]