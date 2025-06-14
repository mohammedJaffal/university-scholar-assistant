package com.fsbmchatbot.fsbmchatbotbackend.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.lang.NonNull;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService; // Important: Use this interface
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import org.springframework.web.util.ContentCachingRequestWrapper;

import java.io.IOException;

@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private static final Logger logger = LoggerFactory.getLogger(JwtAuthenticationFilter.class);

    private final JwtService jwtService;
    private final UserDetailsService userDetailsService; // Inject only the interface

    // Constructor now only takes UserDetailsService
    public JwtAuthenticationFilter(JwtService jwtService, UserDetailsService userDetailsService) {
        this.jwtService = jwtService;
        this.userDetailsService = userDetailsService;
    }
@Override
protected void doFilterInternal(
        @NonNull HttpServletRequest request,
        @NonNull HttpServletResponse response,
        @NonNull FilterChain filterChain) throws ServletException, IOException {

    ContentCachingRequestWrapper wrappedRequest = new ContentCachingRequestWrapper(request);

    try {
        String path = wrappedRequest.getRequestURI();

        if ("/api/auth/register/student".equals(path) || "/api/auth/verify-email".equals(path) || "/api/auth/login".equals(path) || "/api/modules/public".equals(path) || "/api/auth/resend-verification-code".equals(path) || "/api/auth/register/professor".equals(path)) {
            filterChain.doFilter(wrappedRequest, response);
            return;
        }

        String jwt = parseJwt(wrappedRequest); // parser sur wrappedRequest
        if (jwt != null && jwtService.validateJwtToken(jwt)) {
            String username = jwtService.getUserNameFromJwtToken(jwt);

            if (username != null && SecurityContextHolder.getContext().getAuthentication() == null) {
                UserDetails userDetails = this.userDetailsService.loadUserByUsername(username);

                if (jwtService.isTokenValid(jwt, userDetails)) {
                    UsernamePasswordAuthenticationToken authentication =
                            new UsernamePasswordAuthenticationToken(userDetails, null, userDetails.getAuthorities());
                    authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(wrappedRequest));

                    SecurityContextHolder.getContext().setAuthentication(authentication);
                }
            }
        }
    } catch (Exception e) {
        logger.error("Cannot set user authentication: {}", e.getMessage(), e);
    }

    filterChain.doFilter(wrappedRequest, response);
}

   private String parseJwt(HttpServletRequest request) {
    String headerAuth = request.getHeader("Authorization");
    if (headerAuth != null && headerAuth.startsWith("Bearer ")) {
        return headerAuth.substring(7);
    }
    return null;
}

}